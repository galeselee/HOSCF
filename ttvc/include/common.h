#ifndef __INCLUDE_COMMON_H__
#define __INCLUDE_COMMON_H__

#include "tensor.h"
void timefunc(std::function<void(Tensor *, Tensor *, Tensor *, int, int)> f,
              Tensor *, Tensor *, Tensor *, int, int);

#endif