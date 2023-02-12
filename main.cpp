#include <iostream>
#include <functional>
#include <omp.h>

#include "scf.h"
#include "common.h"

#include "mkl_cblas.h"
int threads = 1;

int main(int argc, char **argv) {
    if (argc == 2) {
        threads = std::stoi(argv[1]);
        omp_set_num_threads(threads);
    }

    vint A_shape{16,16,16,16,16,16}; 
    Tensor A(A_shape);
    int ndim = A.ndim;

    for (int ii = 0; ii < A.size; ii++)
        A.data[ii] = randn();

    Tensor U[8];
    for(int ii = 0; ii < ndim; ii++) {
        U[ii].constructor({A.shape[ndim-1-ii]});
        for(int jj = 0; jj < U[ii].size; jj++)
            U[ii].data[jj] = randn();
        U[ii].norm();
    }

    std::function<void(Tensor *, Tensor *, double, uint32_t)> func = scf;

    timescf(func, &A, U, 5.0e-4, 10);
}
