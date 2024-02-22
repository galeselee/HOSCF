#ifndef __SRC_TTVC_INCLUDE_TTVC_H__
#define __SRC_TTVC_INCLUDE_TTVC_H__

#include "tensor.h"

void ttvc_except_dim_mpi(Tensor *A, Tensor *U, double *block_J, int dim0, int dim1);

#endif
