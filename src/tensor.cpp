#include <cstring>
#include <cmath>
#include <algorithm>

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

Tensor::~Tensor() {
    if (data != nullptr) {
        free(data);
    }
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

void Tensor::norm() {
    double sum = 0;
    for (int ii = 0; ii < size; ii++) {
        sum += this->data[ii] * this->data[ii];
    }
    sum = std::sqrt(sum);
    for (int ii = 0; ii < size; ii++) {
        this->data[ii] /= sum;
    }
}

double Tensor::fnorm_range(int begin, int len) {
	double ret = 0.0;
	int end = begin + len;
	for (int ii = begin; ii < end; ii++) {
		ret += this->data[ii] * this->data[ii];
	}
	return std::sqrt(ret);
}

void Tensor::nmul_range(int begin, int len, const double const_num) {
	int end = begin + len;
	for (int ii = begin; ii < end; ii++) {
		this->data[ii] *= const_num;
	}
}







