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
#include "cmdline.h"
using namespace std;

typedef uint32_t u32;
typedef int32_t i32;
typedef float f32;
typedef double f64;


std::chrono::system_clock::time_point tnow() {
    return std::chrono::system_clock::now();
}

void pti(std::chrono::system_clock::time_point time, int iter=-1) {
    auto now_ = tnow();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds >(now_ - time);
    auto pt = time_span.count()/iter;
    std::cout << "[Time] Avg time/ms = "<<std::to_string(pt)<<std::endl;
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

typedef std::vector<u32> vuint;
typedef std::vector<i32> vint;
struct Tensor {
    vint shape;
    u32 size;
    u32 ndim;
    f64 *data = nullptr;
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
    int tensor_stride[ndim];
    tensor_stride[ndim-1] = 1;
    for (int ii = ndim - 2; ii >= 0; ii--) {
        tensor_stride[ii] = tensor_stride[ii+1] * shape[ii+1];
    }

#pragma omp parallel for default(shared)
    for (int ii = 0; ii < shape[adim]; ii++) {
        int idx_ii = ii * tensor_stride[adim];
        for (int jj = 0; jj < shape[ttvc_dim[0]]; jj++) {
            int idx_jj = jj * tensor_stride[ttvc_dim[0]] + idx_ii;
            for (int kk = 0; kk < shape[ttvc_dim[1]]; kk++) {
                int idx_kk = kk * tensor_stride[ttvc_dim[1]] + idx_jj;
                for (int ll = 0; ll < shape[ttvc_dim[2]]; ll++) {
                    int idx_ll = ll * tensor_stride[ttvc_dim[2]] + idx_kk;
                    for (int mm = 0; mm < shape[ttvc_dim[3]]; mm++) {
                        int idx_mm = mm * tensor_stride[ttvc_dim[3]] + idx_ll;
                        for (int nn = 0; nn < shape[ttvc_dim[4]]; nn++) {
                            ret->data[ii] += A->data[idx_mm + nn * tensor_stride[ttvc_dim[4]]] *
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
    int tensor_stride[ndim];
    tensor_stride[ndim-1] = 1;
    for (int ii = ndim -2; ii >= 0; ii--) {
        tensor_stride[ii] = tensor_stride[ii+1] * shape[ii+1];
    }

#pragma omp parallel for default(shared) reduction(+:ret[0])
    for (int ij = 0; ij < shape[0] * shape[1]; ij++) {
        int ii = ij / shape[1];
        int jj = ij % shape[1];
        int idx_jj = jj * tensor_stride[1] + ii * tensor_stride[0];
        for (int kk = 0; kk < shape[2]; kk++) {
            int idx_kk = kk * tensor_stride[2] + idx_jj;
            for (int ll = 0; ll < shape[3]; ll++) {
                int idx_ll = ll * tensor_stride[3] + idx_kk;
                for (int mm = 0; mm < shape[4]; mm++) {
                    int idx_mm = mm * tensor_stride[4] + idx_ll;
                    for (int nn = 0; nn < shape[5]; nn++) {
                        ret[0] += A->data[idx_mm + nn * tensor_stride[5]] * 
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
        // std::cout << "iter = " << iter << ", lambda = " << lambda << ", residual = " << residual
                //  << ", error_delta = " << std::abs(residual - residual_last) << std::endl;
    }
}

int main(int argc, char **argv) {
    cmdline::parser p;
    p.add<int>("ndim", 'n', "Num of dim", false, 6);
    p.add<int>("threads", 't', "Num threads in omp", false, 8);
    p.add<int>("shape", 's', "Shape of tensor. The tensor size will be ndim x shape(only support all dim size equally)", false, 16);
    p.add<int>("repeat", 'r', "Repeat time ", false, 10);
    p.add("help", 0, "print this message");
    p.parse_check(argc, argv);

    u32 omp_threads_num = p.get<int>("threads");
    u32 ndim = p.get<int>("ndim");
    u32 size_one_dim = p.get<int>("shape");
    u32 repeat_num = p.get<int>("repeat");

    omp_set_num_threads(omp_threads_num);

    cout << "[CONFIG] Number dim " << ndim << endl;
    cout << "[CONFIG] OMP set num threads num " << omp_threads_num << std::endl;
    cout << "[CONFIG] Shape " << size_one_dim << std::endl;

    vint shapeA;
    for (u32 ii = 0; ii < ndim; ii++) {
        shapeA.push_back(size_one_dim);
    }
   

    Tensor A;
    for (u32 ii = 0; ii < ndim; ii++) {
        A.shape.push_back(shapeA[ii]);
    }
    A.ndim = ndim;
    A.size = 1;
    for(u32 ii = 0; ii < ndim; ii++)
        A.size *= A.shape[ii];
    
    cout << "[INFO] Tensor A size " << A.size << endl;
    A.data = (f64 *)std::malloc(sizeof(f64) * A.size);

    u32 tensor_stride[ndim];
    tensor_stride[ndim-1] = 1;
    for (i32 ii = ndim-2; ii >= 0; ii--) {
        tensor_stride[ii] = tensor_stride[ii+1] * shapeA[ii+1];
    }
   
    cout << "[INFO] Gen A data"  << endl;
#pragma omp parallel for
    for (u32 ii = 0; ii < A.size; ii++) {
        A.data[ii] = ((ii*17)%11017311)/11013111;
    }

    cout << "[INFO] Gen Rank One Tensors data"  << endl;
  
    Tensor rank_one_tensor_list[ndim];
    for(u32 ii = 0; ii < ndim; ii++) {
        rank_one_tensor_list[ii].ndim = 1;
        rank_one_tensor_list[ii].size = shapeA[ndim-1-ii];
        rank_one_tensor_list[ii].shape.push_back(shapeA[ndim-1-ii]);
        rank_one_tensor_list[ii].data = (double*)std::malloc(sizeof(f64) * rank_one_tensor_list[ii].size);
    }
    auto tt = tnow();
    i32 counter = repeat_num;
    while(counter--) {
        for(u32 ii = 0; ii < ndim; ii++) {
            for(int jj = 0; jj < rank_one_tensor_list[ii].size; jj++)
                rank_one_tensor_list[ii].data[jj] = randn();
            f64 rank_one_tensor_fnorm = fnorm_ptr(rank_one_tensor_list[ii].data, rank_one_tensor_list[ii].size);
            for(int jj = 0; jj < rank_one_tensor_list[ii].size; jj++)
                rank_one_tensor_list[ii].data[jj] /= rank_one_tensor_fnorm;
        }

        als(&A, rank_one_tensor_list, 1e-12, 10);
    }
    pti(tt,repeat_num);

    std::free(A.data);
    for (int ii = 0; ii < ndim; ii++) {
        std::free(rank_one_tensor_list[ii].data);
    }
}
