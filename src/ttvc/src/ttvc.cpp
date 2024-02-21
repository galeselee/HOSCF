#include <iostream>
#include <vector>
#include <omp.h>
#include <assert.h>
#include <cstring>
#include <torch/torch.h>

#include "tensor.h"
#include "ttvc.h"

void ttvc_except_dim_mpi(Tensor *A, Tensor *X, double *block_J, int dim0, int dim1) {
    int ndim = A->ndim;
    int xsize = ndim;
    int adim0 = ndim-1-dim0;
    int adim1 = ndim-1-dim1;
    int block_size = A->shape[adim0] * A->shape[adim1];

    std::vector<int> x_offset(ndim);

    std::vector<torch::Tensor> ttvc_list(A->ndim-1);
    at::IntArrayRef A_shape_torch(&(A->shape[0]), A->ndim);
    ttvc_list[0] = torch::from_blob(A->data, A_shape_torch,torch::kFloat64);
    std::string raw_str = "abcdefghijklmn";
    std::string einsum_str = raw_str.substr(0, A->ndim);
    std::string einsum_ret_str = "";

    int x_offset_counter = 0;
    for (int ii = 0; ii < xsize; ii++) {
        x_offset[ii] = x_offset_counter;
        x_offset_counter += A->shape[A->ndim-ii-1];
        if (dim0 == ii) {
            einsum_ret_str += einsum_str.substr(adim0, 1);
            continue;
        }
        if (dim1 == ii) {
            einsum_ret_str += einsum_str.substr(adim1, 1);
            continue;
        }
        einsum_str += "," + einsum_str.substr(A->ndim-ii-1, 1);
    }
    einsum_str += "->" + einsum_ret_str;

    for (int ii = 0; ii < dim0; ii++) {
        at::IntArrayRef U_shape_torch(&(A->shape[ndim-1-ii]), 1);
        ttvc_list[ii+1] = torch::from_blob(X->data+x_offset[ii], U_shape_torch,torch::kFloat64);
    }
    for (int ii = dim0+1; ii < dim1; ii++) {
        at::IntArrayRef U_shape_torch(&(A->shape[ndim-1-ii]), 1);
        ttvc_list[ii] = torch::from_blob(X->data+x_offset[ii], U_shape_torch,torch::kFloat64);
    }
    for (int ii = dim1+1; ii < xsize; ii++) {
        at::IntArrayRef U_shape_torch(&(A->shape[ndim-1-ii]), 1);
        ttvc_list[ii-1] = torch::from_blob(X->data+x_offset[ii], U_shape_torch,torch::kFloat64);
    }
    auto ttvc_ret = torch::einsum(einsum_str, ttvc_list);
    memcpy(block_J, ttvc_ret.data_ptr(), block_size * sizeof(double));
}