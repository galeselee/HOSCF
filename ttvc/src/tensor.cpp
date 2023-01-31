#include "tensor.h"

Tensor::Tensor() {
    data = nullptr;
}

Tensor::Tensor(vint shape_in) {
    shape = shape_in;
    size = 1;
    ndim = shape.size();
    for (int ii = 0; ii < ndim; ii++) 
        size *= shape[ii];
    data = reinterpret_cast<double *>(malloc(sizeof(double) * size));
    std::memset(data, 0, sizeof(double) * size);
}

void Tensor::constructor(vint shape_in) {
    shape = shape_in;
    size = 1;
    ndim = shape.size();
    for (int ii = 0; ii < ndim; ii++) 
        size *= shape[ii];
    data = reinterpret_cast<double *>(malloc(sizeof(double) * size));
    std::memset(data, 0, sizeof(double) * size);
}