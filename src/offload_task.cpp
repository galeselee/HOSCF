#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <cstring>

#include "ttvc.h"
#include "offload_task.h"


bool compareInterval(std::vector<int> v1, std::vector<int> v2) 
{ 
    return v1[2] > v2[2]; 
} 

int find_min_total_idx(int *total_list, int num_group) {
    int idx = -1;
    int min_total = 1<<29;
    for(int ii = 0; ii < num_group; ii++) {
        if (min_total > total_list[ii]) {
            min_total=total_list[ii];
            idx = ii;
        }
    }
    return idx;
}


void get_avg_arr(std::vector<std::vector<int> >&block_time, int num_group) {
    int block_num = block_time.size();
    int *total_list = (int *)malloc(block_num * sizeof(int));
    memset(total_list, 0, block_num * sizeof(int));

    for (int ii = 0; ii < block_num; ii++) {
        std::vector<std::vector<int>> task_cpu;
        tasks_list.push_back(task_cpu);
    }
    sort(block_time.begin(), block_time.end(), compareInterval);

    for (int ii = 0; ii < block_num; ii++) {
       int min_idx = find_min_total_idx(total_list, num_group);
       tasks_list[min_idx].push_back(block_time[ii]);
       total_list[min_idx] += block_time[ii][2];
    }

}



void offload_task(Tensor *A, Tensor *U) {
    extern int mpi_rank;
    extern int mpi_size;
    int ndim = A->ndim;
    std::vector<std::vector<int>> block_time;
    int block_num = 0;
    for (int ii = 0; ii < ndim; ii++)
        block_num += ii;

    int ttvc_time_buffer[block_num*3];
    int idx_buffer = 0;

    int total_time = 0;
    if (mpi_rank == 0) {
        for (int ii = 0; ii < ndim - 1; ii++) {
            for (int jj = ii+1; jj < ndim; jj++) {
                Tensor block({U[ii].size, U[jj].size});
                auto start = std::chrono::system_clock::now();
                ttvc_except_dim_mpi(A, U, block.data, ii, jj);
                auto end = std::chrono::system_clock::now();
                std::vector<int> elem{ii, jj,
                    static_cast<int>(
                    std::chrono::duration_cast<std::chrono::microseconds>(end-start).count())
                };
                ttvc_time_buffer[idx_buffer++] = elem[0];
                ttvc_time_buffer[idx_buffer++] = elem[1];
                ttvc_time_buffer[idx_buffer++] = elem[2];
            }
        }
    } 
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(ttvc_time_buffer, block_num * 3, MPI_INT, 0, MPI_COMM_WORLD);

    for (int ii = 0; ii < block_num; ii++) {
            std::vector<int> elem{ttvc_time_buffer[ii*3],
                ttvc_time_buffer[ii*3+1],
                ttvc_time_buffer[ii*3+2]}; 
            block_time.push_back(elem);
    }

    get_avg_arr(block_time, mpi_size);


    rank_offset.push_back(0);
    int offset = 0;
    for (int ii = 0; ii < mpi_size-1; ii++) {
        for (int jj = 0; jj < tasks_list[ii].size(); jj++) {
            offset += U[tasks_list[ii][jj][0]].size * U[tasks_list[ii][jj][1]].size;
        }
        rank_offset.push_back(offset);
    }
}