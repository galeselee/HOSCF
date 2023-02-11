#include "scf.h"
#include "ttvc.h"
#include "utils.h"

#include <cstring>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <mpi.h>
#include <vector>
#include <chrono>

extern int size, rank;
extern std::vector<std::vector<std::vector<int>>> tasks_list;
extern std::vector<int> rank_offset;

double cal_lambda(Tensor *A, Tensor *U) {
    vint shape = A->shape;
    int ndim = A->ndim;
    double lambda = 0;
    int scan[ndim];
    scan[ndim-1] = 1;
    for (int ii = ndim-2; ii >= 0; ii--) {
        scan[ii] = scan[ii+1] * shape[ii+1];
    }
    int scan_add[ndim];
    scan_add[0] = 0;
    for (int ii = 1; ii < ndim; ii++) {
        scan_add[ii] = scan_add[ii-1] + shape[A->ndim-ii];
    }
    
#pragma omp parallel for default(shared) reduction(+:lambda)
    for (int ij = 0; ij < shape[0] * shape[1]; ij++) {
        int ii = ij / shape[1];
        int jj = ij % shape[1];
        int idx_ii = ii * scan[0];
        int idx_jj = jj * scan[1] + idx_ii;
        for (int kk = 0; kk < shape[2]; kk++) {
            int idx_kk = kk * scan[2] + idx_jj;
            for (int ll = 0; ll < shape[3]; ll++) {
                int idx_ll = ll * scan[3] + idx_kk;
                for (int uu = 0; uu < shape[4]; uu++) {
                    int idx_uu = uu * scan[4] + idx_ll;
                    for (int tt = 0; tt < shape[5]; tt++) { 
                        lambda += A->data[idx_uu + tt] * U->data[scan_add[0]+tt] * U->data[scan_add[1]+uu]
                                    * U->data[scan_add[2]+ll] * U->data[scan_add[3]+kk]
                                    * U->data[scan_add[4]+jj] * U->data[scan_add[5]+ii];
                    }
                }
            }
        }
    }
// #pragma omp parallel for default(shared) reduction(+:lambda)
//     for (int ij = 0; ij < shape[0] * shape[1]; ij++) {
//         int ii = ij / shape[1];
//         int jj = ij % shape[1];
//         int idx_ii = ii * scan[0];
//         int idx_jj = jj * scan[1] + idx_ii;
//         for (int kk = 0; kk < shape[2]; kk++) {
//             int idx_kk = kk * scan[2] + idx_jj;
//             for (int ll = 0; ll < shape[3]; ll++) {
//                 int idx_ll = ll * scan[3] + idx_kk;
//                 for (int uu = 0; uu < shape[4]; uu++) {
//                     int idx_uu = uu * scan[4] + idx_ll;
//                     for (int tt = 0; tt < shape[5]; tt++) { 
//                         int idx_tt = tt * scan[5] + idx_uu;
//                         for (int rr = 0; rr < shape[6]; rr++) {
//                             int idx_rr = rr * scan[6] + idx_tt;
//                             for (int ee = 0; ee < shape[7]; ee++) {
//                                 int idx_ee = ee  + idx_rr;
//                                 lambda += A->data[idx_ee]
//                                             * U->data[scan_add[0]+ee] * U->data[scan_add[1]+rr]
//                                             * U->data[scan_add[2]+tt] * U->data[scan_add[3]+uu]
//                                             * U->data[scan_add[4]+ll] * U->data[scan_add[5]+kk]
//                                             * U->data[scan_add[6]+jj] * U->data[scan_add[7]+ii];
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
    return lambda;
}

double cal_res(Tensor *J, Tensor *X, double lambda) {
    Tensor w_inter({X->size});

#pragma omp parallel for default(shared)
    for (int ii = 0; ii < J->shape[0]; ii++) {
        int idx = ii * J->shape[1];
        for (int jj = ii+1; jj < J->shape[1]; jj++) {
            w_inter.data[ii] += J->data[idx+jj] * X->data[jj];
        }
    }
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < J->shape[0]; ii++) {
        for (int jj = 0; jj < ii; jj++) {
            w_inter.data[ii] += X->data[jj] * J->data[jj*J->shape[0]+ii];
        }
    }
    double rho = 0.0;

#pragma omp parallel for default(shared) reduction(+:rho)
    for (int ii = 0; ii < X->size; ii++) {
        rho += w_inter.data[ii] * X->data[ii]; // X -> w
    }
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < X->size; ii++) {
        w_inter.data[ii] -= rho * X->data[ii];
    }

    auto res = w_inter.fnorm_range(0, w_inter.size) / 
                (J->fnorm_range(0, J->size)*std::sqrt(2)+std::abs(lambda));
    return res;
}

void fill_J_with_block(Tensor *J, vint shapeA, int x, int y, double *block) {
    int n_J = J->shape[0];
    int x_begin = 0;
    int y_begin = 0;
    int n = shapeA.size();
    int n_x = shapeA[n-1-x];
    int n_y = shapeA[n-1-y];
    for (int i = 0; i < x; i++)
        x_begin += shapeA[n-1-i];
    for (int i = 0; i < y; i++)
        y_begin += shapeA[n-1-i];
    
    for (int i = 0; i < n_x; i++) {
        std::memcpy(J->data + (i+x_begin)*n_J + y_begin, block + i*n_y, sizeof(double) * n_y);
    }
    return ;
}

extern "C" {
	void dsyev_(const char*, const char*, const  int *, double* ,const int *, double *, double *, const int*, int *);
}
void svd_solve(Tensor *J, Tensor *eigvec, double &eig) {
    int n = J->shape[0];
    int lda = n;
    double w[n];
    char V='V';
    char U='L';
    int lwork = 3*n;
    double work[lwork];
    int info;

    dsyev_(&V, &U, &n, J->data, &lda, w, work, &lwork, &info);
    if (info != 0) {
        std::cout << "Error syev @" << __LINE__ << std::endl;
    }
    eig = w[n-1];
    int idx = n - 1;
    if (std::abs(w[0]) > std::abs(w[n-1])) {
        eig = w[0];
        idx = 0;
    }

    memcpy(eigvec->data, &(J->data[idx*n]), sizeof(double)*n);
    return ;
}

void norm_range(double *ptr, int len) {
    double sum = 0.0;
    for (int ii = 0; ii < len; ii++)
        sum += ptr[ii] * ptr[ii];
    double fnorm = std::sqrt(sum);
    for (int ii = 0; ii < len; ii++)
        ptr[ii] /= fnorm;
}

void refact_J(Tensor &block, Tensor &block_mpi, vint shape) {
    int offset = 0;
    for(auto &list : tasks_list) {
        for (auto &task : list) {
            auto ii = task[0];
            auto jj = task[1];
            double *ptr = block_mpi.data + (offset++) * 256;
            fill_J_with_block(&block, shape, ii, jj, ptr);
        }
    }
}

void scf(Tensor *A, Tensor *U, double tol, uint32_t max_iter) {
    int n = A->ndim;
    vint shape = A->shape;
    int iter = 0;
    int n_x = 0;
    int shape_scan[n+1];
    shape_scan[0] = 0;
    for (int ii = 0; ii < n; ii++) {
        n_x += U[ii].size;
        shape_scan[ii+1] = n_x;
    }

    Tensor J({n_x, n_x});
    Tensor J_mpi({n_x, n_x});
    Tensor X({n_x});

    for (int ii = 0; ii < n; ii++) {
        std::memcpy(X.data + shape_scan[ii],
                    U[ii].data, U[ii].size * sizeof(double));
    }

    double lambda = cal_lambda(A, &X);

    while (iter < max_iter) {
        std::memset(J.data, 0, sizeof(double) * J.size);
		MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::system_clock::now(); 
        for (int ii = 0; ii < tasks_list[rank].size(); ii++) {
            int block_ii = tasks_list[rank][ii][0];
            int block_jj = tasks_list[rank][ii][1];
            ttvc_except_dim_mpi(A, &X, J_mpi.data+16*16*(rank_offset[rank]+ii), 
                            block_ii, block_jj);
            norm_range(J_mpi.data+16*16*(rank_offset[rank]+ii), 16*16);
        }
        auto end = std::chrono::system_clock::now(); 
		if (rank == 0)
        std::cout << "ttvc : " \
         << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() / 1000.0
         << "ms" << std::endl; 

        start = std::chrono::system_clock::now();
        MPI_Bcast(J_mpi.data, 6*256, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(J_mpi.data+6*256, 9*256, MPI_DOUBLE, 1, MPI_COMM_WORLD);
        end = std::chrono::system_clock::now();
		if (rank == 0)
        std::cout << "Bcast : " \
         << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() / 1000.0
         << "ms" << std::endl; 

        refact_J(J, J_mpi, shape);

		X.norm();
        auto res = cal_res(&J, &X, lambda);

        if (rank == 0)
            std::cout << iter << "-th scf iteration: lambda is " << lambda << ", residual is " << res << std::endl;

        svd_solve(&J, &X, lambda);

#pragma omp parallel for default(shared)
        for (int ii = 0; ii < n; ii++) {
			X.nmul_range(shape_scan[ii], U[ii].size, 1/X.fnorm_range(shape_scan[ii], U[ii].size));
        }
        iter++;
    }

#pragma omp parallel for
    for (int ii = n-1; ii >= 0; ii--) {
        std::memcpy(U[ii].data, X.data + shape_scan[ii],
                    shape[n-1-ii] * sizeof(double));
    }
    return ;
}
