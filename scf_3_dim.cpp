// bind core
// map i,j -> uniform ij
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <cstddef>
#include <chrono>
#include <tuple>
#include <cstring>
#include <string>
#include <cmath>
#include <map>

//#include "cblas.h"
//#include "lapack.h"
#include "omp.h"

/*MKL*/
//#include "mkl_cblas.h"
//##include "mkl_lapacke.h"
//##include "omp.h"
//##include "mkl.h"
#define NN 2
typedef std::vector<int> vint;
/* struct */
struct Tensor {
    vint shape;
    int size;
    int ndim;
    double *data = nullptr;
};

/* time */
std::chrono::system_clock::time_point tnow() {
    return std::chrono::system_clock::now();
}

int first_iter=0;
int first_ttvc=0;
int ttvc_time = 0;
int svd_time=0;
int init_time=0;
void pti(std::chrono::system_clock::time_point time, std::string info="", int iter=-1) {
    auto now_ = tnow();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds >(now_ - time);
    auto pt = time_span.count();
	if (info == "ttvc") {
		if (iter == 0) {
			first_ttvc = pt;
		}
		else {
			ttvc_time += pt;
		}
	}
	if (info == "svd") {
		if (iter != 0) {
			svd_time += pt;
		}
	}
	if (info == "first_iter" && iter == 0) {
		first_iter = pt;
	}
	if (info == "init") {
	init_time=pt;
	}
	if (info=="total time")
		std::cout << info << " time/ms = "<<std::to_string(pt)<<std::endl;
} 


double randn() {
    double u = ((double)rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double)rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1)
        return randn();
    double c = std::sqrt(-2 * std::log(r) / r);
    return u * c;
}

double fnorm_ptr(double *ptr, int size) {
    double norm = 0;
    for (int i = 0; i < size; i++) {
        norm += ptr[i] * ptr[i];
    }
    return std::sqrt(norm);
}

void Nmul_ptr(double *ptr, double num, int size) {
    for(int ii = 0; ii < size; ii++) {
        ptr[ii] *= num;
    }
}

double cal_lambda(Tensor *A, Tensor *U) {
    vint shape = A->shape;
	int ndim = shape.size();
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
						lambda += A->data[idx_jj + kk]
								  * U->data[scan_add[0]+kk]
								  * U->data[scan_add[1]+jj] * U->data[scan_add[2]+ii];
            }
        }
    return lambda;
}

void ttvc_except_dim(Tensor *A, Tensor *U, Tensor *block_J, int dim0, int dim1) {
    auto shape = A->shape;
    int ndim = A->ndim;
    int a_dim0 = ndim-1-dim0;
    int a_dim1 = ndim-1-dim1;
    block_J->size = A->shape[a_dim0] * A->shape[a_dim1];
    block_J->ndim = 2;
    block_J->shape = {A->shape[a_dim0] * A->shape[a_dim1]};
    block_J->data = (double*)std::malloc(sizeof(double) * block_J->size);
    std::memset(block_J->data, 0, sizeof(double) * block_J->size);
    int dim[ndim-2];
    int cnt = 0;
    for (int ii = 0; ii < A->ndim; ii++) {
        if (ii == a_dim0 || ii == a_dim1) continue;
        dim[cnt++] = ii;
    }
    int scan[ndim];
    scan[ndim-1] = 1;
    for (int ii = ndim-2; ii >= 0; ii--) {
        scan[ii] = scan[ii+1] * shape[ii+1];
    }
    int scan_add[ndim];
    scan_add[0]=0;
    for (int ii = 1; ii < ndim; ii++) {
        scan_add[ii] = scan_add[ii-1] + shape[ndim-ii];
    }

#pragma omp parallel for default(shared)
    for (int ij = 0; ij < shape[a_dim0] * shape[a_dim1]; ij++) {
            int ii = ij / shape[a_dim1];
            int jj = ij % shape[a_dim1];
            int idx_ii = ii * scan[a_dim0];
            int block_idx_ii = ii * shape[a_dim1];
            int idx_jj = jj * scan[a_dim1] + idx_ii;
            int block_idx = block_idx_ii + jj;
            for (int kk = 0; kk < shape[dim[0]]; kk++) {
						block_J->data[block_idx] += 
							A->data[idx_jj + kk*scan[dim[0]]]
							*  U->data[scan_add[ndim-1-dim[0]]+kk];
            }
        }

    return;
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

    memcpy(eigvec->data, &(J->data[idx*n]), 8*n);
//#pragma omp parallel for default(shared)
//    for (int ii = 0; ii < n; ii++) {
//        eigvec->data[ii] = J->data[ii*n+idx];
//    }
    return ;
}

void fill_J_with_block(Tensor *J, vint shapeA, int x, int y, Tensor *block) {
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
        std::memcpy(J->data + (i+x_begin)*n_J + y_begin, block->data + i*n_y, sizeof(double) * n_y);
    }
    return ;
}

double cal_res(Tensor *J, Tensor *X, double lambda) {
    Tensor w_inter;
    w_inter.size = X->size;
    w_inter.ndim = 1;
    w_inter.shape.push_back(X->size);
    w_inter.data = (double*)std::malloc(sizeof(double) * X->size);
    std::memset(w_inter.data, 0, sizeof(double) * X->size);
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

    auto res = fnorm_ptr(w_inter.data, w_inter.size)/(fnorm_ptr(J->data, J->size)*std::sqrt(2)+std::abs(lambda));
    std::free(w_inter.data);
    return res;
}

void scf(Tensor *A, Tensor *U, double tol, int max_iter) {
	auto init_tt = tnow();
    int n = A->ndim;
    vint shape = A->shape;
    int iter = 0;
    int n_j = 0;
    int scan_nj[n+1];
    scan_nj[0] = 0;
    for (int ii = 0; ii < n; ii++) {
        n_j += U[ii].size;
        scan_nj[ii+1] = n_j;
    }

    Tensor J;
    Tensor X;
    J.size = n_j * n_j;
    J.ndim = 2;
    J.shape={n_j, n_j};
    J.data = (double *)std::malloc(sizeof(double) * J.size);
    X.size = n_j;
    X.ndim = 1;
    X.shape = {X.size};
    X.data = (double *)std::malloc(sizeof(double) * X.size);

    for (int ii = 0; ii < n; ii++) {
        std::memcpy(X.data + scan_nj[ii],
                    U[ii].data, U[ii].size * sizeof(double));
    }

    double lambda = cal_lambda(A, &X);
	pti(init_tt,"init");
    while (iter < max_iter) {
		auto first_iter_tt=tnow();
        auto tt = tnow();
        std::memset(J.data, 0, sizeof(double) * J.size);
        for (int ii = 0; ii < n-1; ii++) {
            for (int jj = ii+1; jj < n; jj++) {
                Tensor block_J;
                ttvc_except_dim(A, &X, &block_J, ii, jj);
                Nmul_ptr(block_J.data, 1/((double)n-1), block_J.size); // block / 5
                fill_J_with_block(&J, shape, ii, jj, &block_J);
                std::free(block_J.data);
            }
        }
        pti(tt, "ttvc", iter);

        Nmul_ptr(X.data, 1/ fnorm_ptr(X.data, X.size), X.size);
        auto res = cal_res(&J, &X, lambda);
        
	//std::cout << iter << "-th scf iteration: lambda is " << lambda << ", residual is " << res << std::endl;
       //if (res < tol) {
       //     break;
       // }

        // update X and lambda

		tt = tnow();
        svd_solve(&J, &X, lambda);
		pti(tt, "svd", iter);

#pragma omp parallel for default(shared)
        for (int ii = 0; ii < n; ii++) {
            Nmul_ptr(X.data+scan_nj[ii], 1/fnorm_ptr(X.data+scan_nj[ii], U[ii].size), U[ii].size);
        }
		pti(first_iter_tt, "first_iter", iter);
        iter++;
    }

#pragma omp parallel for
    for (int ii = n-1; ii >= 0; ii--) {
        std::memcpy(U[ii].data, X.data + scan_nj[ii],
                    shape[n-1-ii] * sizeof(double));
    }
    std::free(J.data);
    std::free(X.data);
    return ;
}


int main(int argc, char **argv) {
	if (argc < 2) {
		std::cout << "INFO : use default omp num threads 8" << std::endl;
		omp_set_num_threads(8);
		//mkl_set_num_threads(8);
	}
	else {
		omp_set_num_threads(std::stoi(argv[1]));
		//mkl_set_num_threads(std::stoi(argv[1]));
		std::cout << std::stoi(argv[1])<<std::endl;;
	}

    vint shapeA = {256,256,256}; 

    int ndim = shapeA.size();
    Tensor A;
    for (int ii = 0; ii < ndim; ii++) {
        A.shape.push_back(shapeA[ii]);
    }
    A.ndim = A.shape.size();
    A.size = 1;
    for(int ii = 0; ii < ndim; ii++)
        A.size *= A.shape[ii];
    A.data = (double *)std::malloc(sizeof(double) * A.size);


    int scan[ndim];
    scan[ndim-1] = 1;
    for (int ii = ndim-2; ii >= 0; ii--) {
        scan[ii] = scan[ii+1] * shapeA[ii+1];
    }
   
    for (int ii = 0; ii < shapeA[0]; ii++) {
        int idx_ii = ii * scan[0];
        for (int jj = 0; jj < shapeA[1]; jj++) {
            int idx_jj = jj * scan[1] + idx_ii;
            for (int kk = 0; kk < shapeA[2]; kk++) {
						A.data[idx_jj + kk] = randn();
            }
        }
    }
  
    Tensor U[ndim];
    for(int ii = 0; ii < ndim; ii++) {
        U[ii].ndim = 1;
        U[ii].size = shapeA[ndim-1-ii];
        U[ii].shape.push_back(shapeA[ndim-1-ii]);
        U[ii].data = (double*)std::malloc(sizeof(double) * U[ii].size);
        for(int jj = 0; jj < U[ii].size; jj++)
            U[ii].data[jj] = randn();
        double a = fnorm_ptr(U[ii].data, U[ii].size);
        for(int jj = 0; jj < U[ii].size; jj++)
            U[ii].data[jj] /= a;
    }

	auto tt = tnow();
    scf(&A, U, 5.0e-4, 50);
	pti(tt, "total time");
	std::cout << "ttvc time/ms = "<<std::to_string(ttvc_time)<<std::endl;
	std::cout << "svd time/ms = "<<std::to_string(svd_time)<<std::endl;
	std::cout << "init time/ms = "<<std::to_string(init_time)<<std::endl;
	std::cout << "first iter time/ms = "<<std::to_string(first_iter)<<std::endl;
    std::free(A.data);
    for (int ii = 0; ii < ndim; ii++) {
        std::free(U[ii].data);
    }
}
