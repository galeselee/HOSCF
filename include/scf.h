#ifndef __INCLUDE_SCF_H__
#define __INCLUDE_SCF_H__

#include "tensor.h"

#include <stdint.h>
void scf(Tensor *A, Tensor *U, double tol, uint32_t max_iter);


#endif
