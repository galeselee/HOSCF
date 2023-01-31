#include <iostream>
#include <execution>
#include <cstring>

#include "tensor.h"
#include "ttvc.h"
#include "utils.h"

int threads = 1;

int main(int argc, char **argv) {
    if (argc == 2) {
        threads = std::stoi(argv[1]);
    }

    vint A_shape{16,16,16,16,16,16}; 
    Tensor A(A_shape);
    int ndim = A.ndim;

    for (int ii = 0; ii < A.size; ii++)
        A.data[ii] = randn();

    Tensor U[6];
    for(int ii = 0; ii < ndim; ii++) {
        U[ii].constructor({A.shape[ndim-1-ii]});
        for(int jj = 0; jj < U[ii].size; jj++)
            U[ii].data[jj] = randn();
        U[ii].norm();
    }

    Tensor X({std::reduce(A_shape.begin(), A_shape.end())});
    {
        int fill_size = 0;
        for (int ii = 0; ii < 6; ii++) {
            memcpy(X.data + fill_size, U[ii].data, U[ii].size * sizeof(double));
        }
    }

    Tensor ret;
    ttvc_except_dim(&A, U, &ret, 0, 1);

}