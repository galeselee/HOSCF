#include <iostream>
#include <functional>
#include <omp.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <string>


#include "scf.h"
#include "offload_task.h"
#include "cmdline.h"

#include "mkl_cblas.h"


std::vector<std::vector<std::vector<int> > > tasks_list;
std::vector<int> rank_offset;
i32 mpi_size, mpi_rank;

void init_mpi_vector(Tensor *A, Tensor *U) {
    offload_task(A, U);
}

void printm(std::string header, std::string info, int rank) {
    if (rank == 0) {
        std::cout << header << " " + info << std::endl;
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); // get num of procs
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank); // get my rank      

    printm("[MPI]", "World size " + std::to_string(mpi_size), mpi_rank);

    cmdline::parser p;
    p.add<int>("ndim", 'n', "Num of dim", false, 6);
    // p.add<int>("threads", 't', "Num threads in omp", false, 8);
    // using taskset to control the num of core
    p.add<int>("shape", 's', "Shape of tensor. The tensor size will be ndim x shape(only support all dim size equally)", false, 16);
    p.add<int>("repeat", 'r', "Repeat time ", false, 1);
    p.add<float>("tol", 'e', "Torrance", false, 5e-4);
    p.add("help", 0, "print this message");
    p.parse_check(argc, argv);

    u32 ndim = p.get<int>("ndim");
    u32 size_one_dim = p.get<int>("shape");
    u32 repeat_num = p.get<int>("repeat");
    f32 tol = p.get<float>("tol");

    printm("[CONFIG]", "Number dim " + std::to_string(ndim), mpi_rank);
    printm("[CONFIG]", "Shape " + std::to_string(size_one_dim), mpi_rank);

    vint shapeA;
    for (u32 ii = 0; ii < ndim; ii++) {
        shapeA.push_back(size_one_dim);
    }

    Tensor A(shapeA);

    if (mpi_rank == 0) {
        for (int ii = 0; ii < A.size; ii++)
            A.data[ii] = randn();
        MPI_Send(A.data, A.size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    } else if (mpi_rank == 1) {
        MPI_Recv(A.data, A.size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    Tensor rank_one_tensor_list[ndim];
    Tensor rank_one_tensor_list_bak[ndim];
    for(int ii = 0; ii < ndim; ii++) {
        rank_one_tensor_list[ii].constructor({A.shape[ndim-1-ii]});
        rank_one_tensor_list_bak[ii].constructor({A.shape[ndim-1-ii]});
        if (mpi_rank == 0) {
            for(int jj = 0; jj < rank_one_tensor_list[ii].size; jj++) {
                rank_one_tensor_list_bak[ii].data[jj] = randn();
            }
            rank_one_tensor_list_bak[ii].norm();
            MPI_Send(rank_one_tensor_list_bak[ii].data, rank_one_tensor_list_bak[ii].size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        } else if (mpi_rank == 1) {
            MPI_Recv(rank_one_tensor_list_bak[ii].data, rank_one_tensor_list_bak[ii].size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    init_mpi_vector(&A, rank_one_tensor_list_bak);
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::system_clock::now();
    int counter = repeat_num;
    while(counter--) {
        for(int ii = 0; ii < ndim; ii++) {
            memcpy(rank_one_tensor_list[ii].data, rank_one_tensor_list_bak[ii].data, rank_one_tensor_list_bak[ii].size*sizeof(double));
        }
        scf(&A, rank_one_tensor_list, tol, 10);
    }
    auto end = std::chrono::system_clock::now();

    auto cost_time_total = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() / 1000.0;
    printm("[Time]", "Avg time " + std::to_string(cost_time_total/repeat_num) + " ms", mpi_rank);

    MPI_Finalize();
}
