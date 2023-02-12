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

/*MKL*/
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#include "omp.h"
#include "mkl.h"

std::chrono::system_clock::time_point tnow() {
    return std::chrono::system_clock::now();
}

void pti(std::chrono::system_clock::time_point time, std::string info="", int iter=-1) {
    auto now_ = tnow();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds >(now_ - time);
    auto pt = time_span.count();
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

typedef std::vector<int> vint;
struct Tensor {
    vint shape;
    int size;
    int ndim;
    double *data = nullptr;
};

double fnorm_ptr(double *ptr, int size) {
    double norm = 0;
    for (int i = 0; i < size; i++) {
        norm += ptr[i] * ptr[i];
    }
    return std::sqrt(norm);
}

void ttvc_except_dim(Tensor *A, Tensor *U, Tensor *ret, int dim) {
    auto shape = A->shape;
    int ndim = A->ndim;
    int adim = ndim - 1 - dim;
    ret->size = A->shape[adim];
    ret->ndim = 1;
    ret->shape = {ret->size};
    ret->data = (double *)std::malloc(sizeof(double) * ret->size);
    std::memset(ret->data, 0, sizeof(double) * ret->size);

    int ttvc_dim[5];
    int cnt = 0;
    for (int ii = 0; ii < ndim; ii++) {
        if (ii == adim) continue;
        ttvc_dim[cnt++] = ii;
    }
    int scan[ndim];
    scan[ndim-1] = 1;
    for (int ii = ndim - 2; ii >= 0; ii--) {
        scan[ii] = scan[ii+1] * shape[ii+1];
    }

#pragma omp parallel for default(shared)
    for (int ii = 0; ii < shape[adim]; ii++) {
        int idx_ii = ii * scan[adim];
        for (int jj = 0; jj < shape[ttvc_dim[0]]; jj++) {
            int idx_jj = jj * scan[ttvc_dim[0]] + idx_ii;
            for (int kk = 0; kk < shape[ttvc_dim[1]]; kk++) {
                int idx_kk = kk * scan[ttvc_dim[1]] + idx_jj;
                for (int ll = 0; ll < shape[ttvc_dim[2]]; ll++) {
                    int idx_ll = ll * scan[ttvc_dim[2]] + idx_kk;
                    for (int mm = 0; mm < shape[ttvc_dim[3]]; mm++) {
                        int idx_mm = mm * scan[ttvc_dim[3]] + idx_ll;
                        for (int nn = 0; nn < shape[ttvc_dim[4]]; nn++) {
                            ret->data[ii] += A->data[idx_mm + nn * scan[ttvc_dim[4]]] *
                                       U[ndim-1-ttvc_dim[0]].data[jj] *
                                       U[ndim-1-ttvc_dim[1]].data[kk] *
                                       U[ndim-1-ttvc_dim[2]].data[ll] *
                                       U[ndim-1-ttvc_dim[3]].data[mm] *
                                       U[ndim-1-ttvc_dim[4]].data[nn];
                        }
                    }
                }
            }
        }
    }
}


void ttvc(Tensor *A, Tensor *U, double *ret) {
    vint shape = A->shape;
    int ndim = A->ndim;
    ret[0] = 0.0f;
    int scan[ndim];
    scan[ndim-1] = 1;
    for (int ii = ndim -2; ii >= 0; ii--) {
        scan[ii] = scan[ii+1] * shape[ii+1];
    }

#pragma omp parallel for default(shared) reduction(+:ret[0])
    for (int ij = 0; ij < shape[0] * shape[1]; ij++) {
        int ii = ij / shape[1];
        int jj = ij % shape[1];
        int idx_jj = jj * scan[1] + ii * scan[0];
        for (int kk = 0; kk < shape[2]; kk++) {
            int idx_kk = kk * scan[2] + idx_jj;
            for (int ll = 0; ll < shape[3]; ll++) {
                int idx_ll = ll * scan[3] + idx_kk;
                for (int mm = 0; mm < shape[4]; mm++) {
                    int idx_mm = mm * scan[4] + idx_ll;
                    for (int nn = 0; nn < shape[5]; nn++) {
                        ret[0] += A->data[idx_mm + nn * scan[5]] * 
                                U[5].data[ii] * U[4].data[jj] *
                                U[3].data[kk] * U[2].data[ll] *
                                U[1].data[mm] * U[0].data[nn];
                    }
                }
            }
        }
    }
}

void Nmul_ptr(double *ptr, double num, int size) {
    for(int ii = 0; ii < size; ii++) {
        ptr[ii] *= num;
    }
}

void als(Tensor *A, Tensor *U, double tol, int max_iter) {
    int ndim = A->ndim;
    vint shape = A->shape;
    double residual = 0.0;
    double residual_last = 2 * tol;
    int iter = 0;
    double lambda = 0.0;
    double AF = fnorm_ptr(A->data, A->size);
    while (std::abs(residual - residual_last) > tol && iter < max_iter) {
        for (int ii = 0; ii < ndim; ii++) {
            Tensor ret;
            ttvc_except_dim(A, U, &ret, ii);
            Nmul_ptr(ret.data, 1 / fnorm_ptr(ret.data, ret.size), ret.size);
            std::memcpy(U[ii].data, ret.data, ret.size * sizeof(double));
        }
        ttvc(A, U, &lambda);
        residual_last = residual;
        residual = std::sqrt(1 - (lambda * lambda) / (AF * AF));
        iter ++;
        std::cout << "iter = " << iter << ", lambda = " << lambda << ", residual = " << residual
                 << ", error_delta = " << std::abs(residual - residual_last) << std::endl;
    }
}

int main(int argc, char **argv) {
	if (argc < 2) {
		std::cout << "INFO : use default omp num threads 8" << std::endl;
		omp_set_num_threads(8);
	}
	else {
		omp_set_num_threads(std::stoi(argv[1]));
	}

    vint shapeA;
    for (int ii = 0; ii < 6; ii++) {
        shapeA.push_back(16);
    }

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
   
    for (int ii = 0; ii < A.size; ii++) {
        A.data[ii] = randn();
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
    als(&A, U, 1e-12, 10);
	pti(tt, "total time");
    std::free(A.data);
    for (int ii = 0; ii < ndim; ii++) {
        std::free(U[ii].data);
    }
}
