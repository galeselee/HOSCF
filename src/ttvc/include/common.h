#ifndef __SRC_TTVC_INCLUDE_COMMON_H__
#define __SRC_TTVC_INCLUDE_COMMON_H__

#include "tensor.h"
void timettvc(std::function<void(Tensor *, Tensor *, Tensor *, int, int)> f,
              Tensor *, Tensor *, Tensor *, int, int);

void timescf(std::function<void(Tensor *, Tensor *, double, uint32_t)> f,
              Tensor *, Tensor *, double, uint32_t);

#endif