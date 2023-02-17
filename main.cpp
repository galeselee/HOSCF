#include <iostream>
#include <functional>
#include <omp.h>
#include <mpi.h>
#include <vector>
#include <cmath>


#include "scf.h"
#include "common.h"
#include "offload_task.h"

#include "mkl_cblas.h"

int threads = 1;
std::vector<std::vector<std::vector<int> > > tasks_list;
std::vector<int> rank_offset;
int size, rank;

int NDIM = 4;

void init_mpi_vector(Tensor *A, Tensor *U) {
    offload_task(A, U);
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
    } else if (argc == 3) {
        threads = std::stoi(argv[2]);
        omp_set_num_threads(threads);
        NDIM = std::stoi(argv[1]);
    }
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get num of procs
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my rank 

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

    if (rank == 0) {
        for (int ii = 0; ii < A.size; ii++)
            A.data[ii] = randn();
        MPI_Send(A.data, A.size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1){
        MPI_Recv(A.data, A.size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    Tensor U[NDIM];
    for(int ii = 0; ii < ndim; ii++) {
        U[ii].constructor({A.shape[ndim-1-ii]});
        if (rank == 0) {
            for(int jj = 0; jj < U[ii].size; jj++)
                U[ii].data[jj] = randn();
            U[ii].norm();
            MPI_Send(U[ii].data, U[ii].size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        } else if (rank == 1) {
            MPI_Recv(U[ii].data, U[ii].size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    init_mpi_vector(&A, U);

    std::function<void(Tensor *, Tensor *, double, uint32_t)> func = scf;

    MPI_Barrier(MPI_COMM_WORLD);
    timescf(func, &A, U, 5.0e-4, 10);
    MPI_Finalize();
}
