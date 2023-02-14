#include <iostream>
#include <functional>
#include <omp.h>

#include "scf.h"
#include "common.h"

int threads = 1;

int NDIM = 4;

int main(int argc, char **argv) {
    if (argc == 2) {
        threads = std::stoi(argv[1]);
        omp_set_num_threads(threads);
    } else if (argc == 3) {
        threads = std::stoi(argv[2]);
        omp_set_num_threads(threads);
        NDIM = std::stoi(argv[1]);
    }

    vint A_shape;
    if (NDIM == 8) {
        for (int ii = 0; ii < 8; ii++) {
            A_shape.push_back(8);
        }
    }
    else if (NDIM == 7) {
        for (int ii = 0; ii < 3; ii++)
            A_shape.push_back(16);
        for (int ii = 0; ii < 4; ii++)
            A_shape.push_back(8);
    } else if (NDIM == 6) {
        for (int ii = 0; ii < 6; ii++) {
            A_shape.push_back(16);
        }
    } else if (NDIM == 5) {
        for (int ii = 0; ii < 4; ii++) {
            A_shape.push_back(32);
        }
        A_shape.push_back(16);
    } else {
        for (int ii = 0; ii < 4; ii++) {
            A_shape.push_back(64);
        }
    } 
    Tensor A(A_shape);
    int ndim = A.ndim;

    for (int ii = 0; ii < A.size; ii++)
        A.data[ii] = randn();

    Tensor U[NDIM];
    for(int ii = 0; ii < ndim; ii++) {
        U[ii].constructor({A.shape[ndim-1-ii]});
        for(int jj = 0; jj < U[ii].size; jj++)
            U[ii].data[jj] = randn();
        U[ii].norm();
    }

    std::function<void(Tensor *, Tensor *, double, uint32_t)> func = scf;

    timescf(func, &A, U, 5.0e-4, 10);
}
