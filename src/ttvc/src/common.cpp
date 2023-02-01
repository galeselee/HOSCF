#include <chrono>
#include <functional>
#include <iostream>

#include "common.h"

void timefunc(std::function<void(Tensor *, Tensor *, Tensor *, int, int)> f,
              Tensor *A, Tensor *U, Tensor *ret, int dim0, int dim1) {
    auto start = std::chrono::system_clock::now();
    f(A, U, ret, dim0, dim1);
    auto end = std::chrono::system_clock::now();
    std::cout << "time : " \
         << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() / 1000.0
         << "ms" << std::endl;
}