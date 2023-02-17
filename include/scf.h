#ifndef __INCLUDE_SCF_H__
#define __INCLUDE_SCF_H__

#include "tensor.h"

#include <stdint.h>
void scf(Tensor *A, Tensor *U, double tol, uint32_t max_iter);

extern int NDIM;
extern int threads;
extern std::vector<std::vector<std::vector<int> > > tasks_list;
extern std::vector<int> rank_offset;
extern int size, rank;

#endif
