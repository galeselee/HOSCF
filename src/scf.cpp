#include "scf.h"
#include "ttvc.h"
#include "utils.h"
#include <omp.h>

#include <cstring>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <mpi.h>
#include <chrono>

#include <torch/torch.h>

double cal_lambda(Tensor *A, Tensor *U) {
    vint shape = A->shape;
    int ndim = A->ndim;
    double lambda = 0;
    std::vector<torch::Tensor> ttvc_list(ndim+1);
    at::IntArrayRef A_shape_torch(&(A->shape[0]), A->ndim);
    ttvc_list[0] = torch::from_blob(A->data, A_shape_torch,torch::kFloat64);
    std::string raw_str = "abcdefghijklmn";
    std::string einsum_str = raw_str.substr(0, A->ndim);
    for (int ii = 0; ii < A->ndim; ii++) {
        einsum_str += "," + einsum_str.substr(A->ndim-ii-1, 1);
    }

    for (int ii = 0; ii < A->ndim; ii++) {
        at::IntArrayRef U_shape_torch(&(U[ii].shape[0]), U[ii].ndim);
        ttvc_list[ii+1] = torch::from_blob(U[ii].data, U_shape_torch,torch::kFloat64);
    }
    auto ttvc_ret = torch::einsum(einsum_str, ttvc_list);
    lambda = ((double*)ttvc_ret.data_ptr())[0];

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
    int ndim = shape.size();
    std::vector<int> offset{0};
    int offset_idx = 0;
    for(int ii = 1; ii < tasks_list[0].size(); ii++) {
        offset.push_back(shape[ndim-1-tasks_list[0][ii-1][0]] * 
                         shape[ndim-1-tasks_list[0][ii-1][1]] + offset[ii-1]);
    }
    offset.push_back(shape[ndim-1-tasks_list[0][tasks_list[0].size()-1][0]] * 
                     shape[ndim-1-tasks_list[0][tasks_list[0].size()-1][1]] + offset[tasks_list[0].size()-1]);
    int idx_bias = tasks_list[0].size();

    for (int ii = 1; ii < tasks_list[1].size(); ii++) {
        offset.push_back(shape[ndim-1-tasks_list[1][ii-1][0]] * 
                         shape[ndim-1-tasks_list[1][ii-1][1]] + offset[ii-1+idx_bias]);
    }


    for(auto &list : tasks_list) {
        for (auto &task : list) {
            auto ii = task[0];
            auto jj = task[1];
            double *ptr = block_mpi.data + offset[offset_idx++];
            fill_J_with_block(&block, shape, ii, jj, ptr);
        }
    }
}

void scf(Tensor *A, Tensor *U, double tol, uint32_t max_iter) {
    int ndim = A->ndim;
    vint shape = A->shape;
    int iter = 0;
    int n_x = 0;
    int shape_scan[ndim+1];
    shape_scan[0] = 0;
    for (int ii = 0; ii < ndim; ii++) {
        n_x += U[ii].size;
        shape_scan[ii+1] = n_x;
    }

    Tensor J({n_x, n_x});
    Tensor J_mpi({n_x, n_x});
    Tensor X({n_x});

    for (int ii = 0; ii < ndim; ii++) {
        std::memcpy(X.data + shape_scan[ii],
                    U[ii].data, U[ii].size * sizeof(double));
    }

    int size_rank0 = 0;
    int size_rank1 = 0;
    for (int ii = 0; ii < tasks_list[0].size(); ii++) {
        int u_ii = tasks_list[0][ii][0];
        int u_jj = tasks_list[0][ii][1]; 
        size_rank0 += U[u_ii].size * U[u_jj].size;
    }
    for (int ii = 0; ii < tasks_list[1].size(); ii++) {
        int u_ii = tasks_list[1][ii][0];
        int u_jj = tasks_list[1][ii][1]; 
        size_rank1 += U[u_ii].size * U[u_jj].size;
    }

    double lambda = cal_lambda(A, U);

    while (iter < max_iter) {
        std::memset(J.data, 0, sizeof(double) * J.size);
        int store_offset = 0;
        for (int ii = 0; ii < tasks_list[mpi_rank].size(); ii++) {
            int block_ii = tasks_list[mpi_rank][ii][0];
            int block_jj = tasks_list[mpi_rank][ii][1];
            ttvc_except_dim_mpi(A, &X, J_mpi.data+rank_offset[mpi_rank]+store_offset, 
                            block_ii, block_jj);
            norm_range(J_mpi.data+rank_offset[mpi_rank]+store_offset,
                       U[block_ii].size * U[block_jj].size);
            store_offset += U[block_ii].size * U[block_jj].size;
        }
        MPI_Bcast(J_mpi.data, size_rank0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(J_mpi.data + size_rank0, size_rank1, MPI_DOUBLE, 1, MPI_COMM_WORLD);

        refact_J(J, J_mpi, shape);

		X.norm();
        auto res = cal_res(&J, &X, lambda);

        std::cout << iter << "-th scf iteration: lambda is " << lambda << ", residual is " << res << std::endl;

        svd_solve(&J, &X, lambda);

#pragma omp parallel for default(shared)
        for (int ii = 0; ii < ndim; ii++) {
			X.nmul_range(shape_scan[ii], U[ii].size, 1/X.fnorm_range(shape_scan[ii], U[ii].size));
        }
        iter++;
    }

#pragma omp parallel for
    for (int ii = ndim-1; ii >= 0; ii--) {
        std::memcpy(U[ii].data, X.data + shape_scan[ii],
                    shape[ndim-1-ii] * sizeof(double));
    }
    return ;
}
