#ifndef __INCLUDE_SCF_H__
#define __INCLUDE_SCF_H__

#include "tensor.h"

extern std::vector<std::vector<std::vector<int> > > tasks_list;
extern std::vector<int> rank_offset;

extern int32_t mpi_size;
extern int32_t mpi_rank;
void scf(Tensor *A, Tensor *U, double tol, uint32_t max_iter);

#endif
