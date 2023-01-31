#ifndef __INCLUDE_TENSOR_H__
#define __INCLUDE_TENSOR_H__

#include <string>

#include "utils.h"

class Tensor {
public:
    vint shape;
    int size;
    int ndim;
    double *data;
    Tensor();
    Tensor(vint shape_in);
    ~Tensor();
    void constructor(vint shape_in);
    void norm();
};

#endif