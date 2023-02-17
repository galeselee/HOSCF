#ifndef __INCLUDE_OFFLOAD_TASK_H__
#define __INCLUDE_OFFLOAD_TASK_H__

#include <vector>
extern std::vector<std::vector<std::vector<int> > > tasks_list;
extern std::vector<int> rank_offset;

#include "tensor.h"

void offload_task(Tensor *A, Tensor *U);

#endif