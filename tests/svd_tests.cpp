#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <cstddef>
#include <chrono>
#include <tuple>
#include <cstring>
#include <string>
#include <cmath>
#include <map>


/*MKL*/
#include "mkl_cblas.h"
#include "mkl_lapacke.h"


void svd_solve() {
// dsyev
    char jobz = 'V';
    char uplo = 'U';
    lapack_int lda=3,n=3;
    double a[9] = {3,2,4,
		           2,0,2,
				   4,2,3};
    double *w = (double *)malloc(sizeof(double) * 9);
	auto info=LAPACKE_dsyev(LAPACK_ROW_MAJOR, jobz, uplo, n, a, lda, w);    
	if (info != 0) {
		std::cout << "Err" << std::endl;
		exit(1);
	}

	std::cout << "EigenValue = ";
	for (int ii = 0; ii < 3; ii++) {
		std::cout << " " << w[ii];
	}
	std::cout << std::endl;
	std::cout << "============ Eigen Vector=========" << std::endl;
	for (int ii = 0; ii < 3; ii++) {
		for (int jj = 0; jj < 3; jj++) {
			std::cout << a[ii*3+jj];
		}
		std::cout << std::endl;
	}
}

int main() {
	svd_solve();
}
