#ifndef __SRC_TTVC_INCLUDE_TENSOR_H__
#define __SRC_TTVC_INCLUDE_TENSOR_H__

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
	double fnorm_range(int , int);
	void nmul_range(int, int, const double);
};

#endif
