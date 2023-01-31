#include <iostream>
#include <cstring>

#include "tensor.h"
#include "ttvc.h"
#include "utils.h"
#include "common.h"

int threads = 1;

int main(int argc, char **argv) {
    if (argc == 2) {
        threads = std::stoi(argv[1]);
    }

    vint A_shape{24,24,24,24,24,24}; 
    Tensor A(A_shape);
    int ndim = A.ndim;

    for (int ii = 0; ii < A.size; ii++)
        A.data[ii] = randn();

    Tensor U[6];
    for(int ii = 0; ii < ndim; ii++) {
        U[ii].constructor({A.shape[ndim-1-ii]});
        for(int jj = 0; jj < U[ii].size; jj++)
            U[ii].data[jj] = randn();
        U[ii].norm();
    }

	int sum_shape = 0;
	for (int ii = 0; ii < A_shape.size(); ii++)
		sum_shape += A_shape[ii];

    Tensor X({sum_shape});
    {
        int fill_size = 0;
        for (int ii = 0; ii < 6; ii++) {
            memcpy(X.data + fill_size, U[ii].data, U[ii].size * sizeof(double));
        }
    }

    Tensor ret;
    std::function<void(Tensor *, Tensor *, Tensor *, int, int)> func = 
        ttvc_except_dim;
    timefunc(func, &A, &X, &ret, 0, 1);

}
