#ifndef __SRC_TTVC_INCLUDE_TTVC_H__
#define __SRC_TTVC_INCLUDE_TTVC_H__

#include "tensor.h"
#include "util.h"

extern int threads;
void ttvc_except_dim(Tensor *A, Tensor *U, Tensor *block_J, int dim0, int dim1);

#endif