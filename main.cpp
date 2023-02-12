#include <iostream>
#include <functional>
#include <omp.h>
#include <mpi.h>
#include <vector>
#include <cmath>

#include "scf.h"
#include "common.h"

#include "mkl_cblas.h"

int threads = 1;
std::vector<std::vector<std::vector<int> > > tasks_list;
std::vector<int> rank_offset;

void init_mpi_vector() {
    tasks_list.push_back({{0, 1}, {0, 2}, {2, 3}, {0, 3}, {0, 4}, {2, 4}});
    tasks_list.push_back({{0, 5}, {1, 2}, {2, 5}, {3, 4}, {3, 5}, {1, 3}, {1, 4}, {1, 5}, {4, 5}});
    rank_offset.push_back(0);
    rank_offset.push_back(6);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get num of procs
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my rank      
    std::cout << rank << " " << size << std::endl;

    init_mpi_vector();
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
    MPI_Finalize();
}
