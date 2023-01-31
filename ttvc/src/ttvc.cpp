#include <iostream>
#include <vector>
#include <omp.h>
#include <assert.h>
#include <cstring>

#include "tensor.h"
#include "ttvc.h"

void ttvc_except_dim(Tensor *A, Tensor *U, Tensor *block_J, int dim0, int dim1) {
    auto shape = A->shape;
    auto ndim = A->ndim;
    assert(dim0 < ndim);
    assert(dim1 < ndim);
    int a_dim0 = ndim-1-dim0;
    int a_dim1 = ndim-1-dim1;
    
    block_J->constructor({A->shape[a_dim0], A->shape[a_dim1]});

    int dim[ndim-2]; // C++ 11
    int cnt = 0;

    for (int ii = 0; ii < ndim; ii++) {
        if (ii == a_dim0 || ii == a_dim1) continue;
        dim[cnt++] = ii;
    }

    int A_stride[ndim];
    A_stride[ndim-1] = 1;
    for (int ii = ndim-2; ii >= 0; ii--) {
        A_stride[ii] = A_stride[ii+1] * shape[ii+1];
    }

    int Ui_index[ndim];
    Ui_index[0]=0;
    for (int ii = 1; ii < ndim; ii++) {
        Ui_index[ii] = Ui_index[ii-1] + shape[ndim-ii];
    }

    int outer_loop = shape[a_dim0] * shape[a_dim1];
    int loop_index0 = shape[dim[0]];
    int loop_index1 = shape[dim[1]];
    int loop_index2 = shape[dim[2]];
    int loop_index3 = shape[dim[3]];
    int stride0 = A_stride[dim[0]];
    int stride1 = A_stride[dim[1]];
    int stride2 = A_stride[dim[2]];
    int stride3 = A_stride[dim[3]];
    int index0 = Ui_index[ndim-1-dim[0]];
    int index1 = Ui_index[ndim-1-dim[1]];
    int index2 = Ui_index[ndim-1-dim[2]];
    int index3 = Ui_index[ndim-1-dim[3]];

    omp_set_num_threads(threads);

//#pragma omp parallel for default(shared)
    for (int ij = 0; ij < outer_loop; ij++) {
    std::cout << ij << std::endl;
        int ii = ij / shape[a_dim1];
        int jj = ij % shape[a_dim1];
        int idx_ii = ii * A_stride[a_dim0];
        int idx_jj = jj * A_stride[a_dim1] + idx_ii;
        int block_idx = ii * shape[a_dim1] + jj;
        for (int kk = 0; kk < loop_index0; kk++) {
            int idx_kk = kk * stride0 + idx_jj;
            for (int ll = 0; ll < loop_index1; ll++) {
                int idx_ll = ll * stride1 + idx_kk;
                for (int uu = 0; uu < loop_index2; uu++) {
                    int idx_uu = uu * stride2 + idx_ll;
                    for (int tt = 0; tt < loop_index3; tt++) {
                        block_J->data[block_idx] += 
                            A->data[idx_uu + tt * stride3] * 
                            U->data[Ui_index[index3]+tt] * 
                            U->data[Ui_index[index2]+uu] * 
                            U->data[Ui_index[index1]+ll] * 
                            U->data[Ui_index[index0]+kk];
                    }
                }
            }
        }
    break;
    }

    std::cout << "b" << std::endl;
    return;
}