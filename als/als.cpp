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

#include <torch/torch.h>
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

typedef std::vector<long int> vint;
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
    int adim = A->ndim-1-dim;
    int u_size = A->ndim;

    ret->size = A->shape[adim];
    ret->ndim = 1;
    ret->shape = {ret->size};
    ret->data = (double *)std::malloc(sizeof(double) * ret->size);

    vector<torch::Tensor> ttvc_list(A->ndim);
    at::IntArrayRef A_shape_torch(&(A->shape[0]), A->ndim);
    ttvc_list[0] = torch::from_blob(A->data, A_shape_torch,torch::kFloat64);
    string raw_str = "abcdefghijklmn";
    string einsum_str = raw_str.substr(0, A->ndim);
    string einsum_ret_str = "";

    for (int ii = 0; ii < u_size; ii++) {
        if (dim == ii) {
            einsum_ret_str += "->" + einsum_str.substr(adim, 1);
            continue;
        }
        einsum_str += "," + einsum_str.substr(A->ndim-ii-1, 1);
    }

    einsum_str += einsum_ret_str;

    for (int ii = 0; ii < dim; ii++) {
        at::IntArrayRef U_shape_torch(&(U[ii].shape[0]), U[ii].ndim);
        ttvc_list[ii+1] = torch::from_blob(U[ii].data, U_shape_torch,torch::kFloat64);
    }
    for (int ii = dim+1; ii < u_size; ii++) {
        at::IntArrayRef U_shape_torch(&(U[ii].shape[0]), U[ii].ndim);
        ttvc_list[ii] = torch::from_blob(U[ii].data, U_shape_torch,torch::kFloat64);
    }
    auto ttvc_ret = torch::einsum(einsum_str, ttvc_list);
    memcpy(ret->data, ttvc_ret.data_ptr(), ret->size * sizeof(double));
}


void ttvc(Tensor *A, Tensor *U, double *ret) {
    vector<torch::Tensor> ttvc_list(A->ndim+1);
    at::IntArrayRef A_shape_torch(&(A->shape[0]), A->ndim);
    ttvc_list[0] = torch::from_blob(A->data, A_shape_torch,torch::kFloat64);
    string raw_str = "abcdefghijklmn";
    string einsum_str = raw_str.substr(0, A->ndim);
    for (int ii = 0; ii < A->ndim; ii++) {
        einsum_str += "," + einsum_str.substr(A->ndim-ii-1, 1);
    }

    for (int ii = 0; ii < A->ndim; ii++) {
        at::IntArrayRef U_shape_torch(&(U[ii].shape[0]), U[ii].ndim);
        ttvc_list[ii+1] = torch::from_blob(U[ii].data, U_shape_torch,torch::kFloat64);
    }
    auto ttvc_ret = torch::einsum(einsum_str, ttvc_list);
    ret[0] = ((double*)ttvc_ret.data_ptr())[0];
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
    // while (std::abs(residual - residual_last) > tol && iter < max_iter) {
    while (iter < max_iter) {
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
    cmdline::parser p;
    p.add<int>("ndim", 'n', "Num of dim", false, 6);
    // p.add<int>("threads", 't', "Num threads in omp", false, 8);
    // using taskset to control the num of core
    p.add<int>("shape", 's', "Shape of tensor. The tensor size will be ndim x shape(only support all dim size equally)", false, 16);
    p.add<int>("repeat", 'r', "Repeat time ", false, 1);
    p.add("help", 0, "print this message");
    p.parse_check(argc, argv);

    // u32 omp_threads_num = p.get<int>("threads");
    u32 ndim = p.get<int>("ndim");
    u32 size_one_dim = p.get<int>("shape");
    u32 repeat_num = p.get<int>("repeat");

    // omp_set_num_threads(omp_threads_num);

    cout << "[CONFIG] Number dim " << ndim << endl;
    // cout << "[CONFIG] OMP set num threads num " << omp_threads_num << std::endl;
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
