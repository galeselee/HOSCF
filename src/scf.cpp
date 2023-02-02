#include "scf.h"

void scf(Tensor *A, Tensor *U, double tol, uint32_t max_iter) {
    int ndim = A->ndim;
    vint shape = A->shape;
    int iter = 0;
    int n_x = 0;
    int shape_scan[7];
    shape_scan[0] = 0;
    for (int ii = 0; ii < n; ii++) {
        n_x += U[ii].size;
        shape_scan[ii+1] = n_x;
    }

    Tensor J({n_x, n_x});
    Tensor X({n_x});

    for (int ii = 0; ii < n; ii++) {
        std::memcpy(X.data + shape_scan[ii],
                    U[ii].data, U[ii].size * sizeof(double));
    }

    double lambda = cal_lambda(A, &X);

    while (iter < max_iter) {
        std::memset(J.data, 0, sizeof(double) * J.size);
        for (int ii = 0; ii < n-1; ii++) {
            for (int jj = ii+1; jj < n; jj++) {
                Tensor block_J;
                ttvc_except_dim(A, &X, &block_J, ii, jj);
				block_J.norm();
                fill_J_with_block(&J, shape, ii, jj, &block_J);
            }
        }

		X.norm();
        auto res = cal_res(&J, &X, lambda);

        std::cout << iter << "-th scf iteration: lambda is " << lambda << ", residual is " << res << std::endl;
        if (res < tol) {
             break;
         }

        svd_solve(&J, &X, lambda);

#pragma omp parallel for default(shared)
        for (int ii = 0; ii < n; ii++) {
			X.nmul_range(shape_scan[ii], U[ii].size, 1/X.fnorm_range(shape_scan[ii], U[ii].size));
        }
        iter++;
    }

#pragma omp parallel for
    for (int ii = n-1; ii >= 0; ii--) {
        std::memcpy(U[ii].data, X.data + scan_nj[ii],
                    shape[n-1-ii] * sizeof(double));
    }
    return ;
}
