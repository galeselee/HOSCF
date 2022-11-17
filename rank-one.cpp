
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

typedef std::vector<int> vint;

std::map<double *, int> ref_count;
/* struct */
struct Tensor {
    vint shape;
    int size;
    int ndim;
    double *data = nullptr;
    void copy(Tensor &b) {
        if(ref_count[this->data] == 0 && this->data != nullptr) {
            std::free(this->data);
        }
        shape = {};
        for(int ii = 0; ii < b.shape.size(); ii++) {
            int t = b.shape[ii];
            this->shape.push_back(t);
        }
        this->size = b.size;
        this->ndim = b.ndim;
        this->data = (double *)std::malloc(sizeof(double) * b.size);
        ref_count[this->data] = 1;
        std::memcpy(this->data, b.data, sizeof(double) * this->size);
    }
    Tensor& operator=(const Tensor& b)
    {
        this->size = b.size;
        this->shape = {};
        for(int ii = 0; ii < b.shape.size(); ii++) {
            int t = b.shape[ii];
            this->shape.push_back(t);
        }
        this->ndim = b.ndim;
        if(this->data!=nullptr) ref_count[this->data]--;
        if (ref_count[this->data] == 0 && this->data != nullptr) { 
            ref_count.erase(this->data);
        }
        this->data = b.data;
        if (this->data != nullptr) { ref_count[this->data]++; }
        return *this;
    }
    Tensor(const Tensor& b)//复制构造函数
    {
        *this = b;
    }
    Tensor() {
        this->data = nullptr;
    }
    ~Tensor() {
        if(this->data!=nullptr){
            ref_count[this->data]--;
            if (ref_count[this->data] == 0)  {
                std::free(this->data);
                ref_count.erase(this->data);
            }
        }
    }
};

/* time */
std::chrono::system_clock::time_point tnow() {
    return std::chrono::system_clock::now();
}

void pti(std::chrono::system_clock::time_point time) {
    auto now_ = tnow();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds >(now_ - time);
    auto pt = time_span.count();
    std::cout<<"time/ms = "<<std::to_string(pt)<<std::endl;
} 


double randn() {
    double u = ((double)rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double)rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1)
        return randn();
    double c = std::sqrt(-2 * std::log(r) / r);
    return u * c;
}

double fnorm(Tensor &A) {
    int n = A.size;
    double norm = 0;
    for (size_t i = 0; i < n; i++) {
        norm += A.data[i] * A.data[i];
    }
    return std::sqrt(norm);
}

void Nmul(Tensor &A, double num) {
    for(int ii = 0; ii < A.size; ii++) {
        A.data[ii] *= num;
    }
}

double cal_lambda(Tensor &A, Tensor &U) {
    int n = A.ndim;
    vint shape = A.shape;
    int lambda = 0;
    int scan[6];
    scan[5] = 1;
    for (int ii = 4; ii >= 0; ii--) {
        scan[ii] = scan[ii+1] * shape[ii+1];
    }
    int scan_add[6] = {0};
    for (int ii = 1; ii < 6; ii++) {
        scan_add[ii] = scan_add[ii-1] + shape[ii-1];
    }

    for (int ii = 0; ii < shape[0]; ii++) {
        int idx_ii = ii * scan[0];
        for (int jj = 0; jj < shape[1]; jj++) {
            int idx_jj = jj * scan[1] + idx_ii;
            for (int kk = 0; kk < shape[2]; kk++) {
                int idx_kk = kk * scan[2] + idx_jj;
                for (int ll = 0; ll < shape[3]; ll++) {
                    int idx_ll = ll * scan[3] + idx_kk;
                    for (int uu = 0; uu < shape[4]; uu++) {
                        int idx_uu = uu * scan[4] + idx_ll;
                        for (int tt = 0; tt < shape[5]; tt++) {
                            lambda += A.data[idx_uu + tt] * U.data[scan_add[0]+ii] * U.data[scan_add[1]+jj]
                                      * U.data[scan_add[2]+kk] * U.data[scan_add[3]+ll]
                                      * U.data[scan_add[4]+uu] * U.data[scan_add[5]+tt];
                        }
                    }
                }
            }
        }
    }
    return lambda;
}

Tensor ttvc_except_dim(Tensor &A, Tensor &U, int dim0, int dim1) {
    auto shape = A.shape;
    Tensor block_J;
    block_J.size = A.shape[dim0] * A.shape[dim1];
    block_J.ndim = 2;
    block_J.shape = {A.shape[dim0] * A.shape[dim1]};
    block_J.data = (double*)std::malloc(sizeof(double) * block_J.size);
    ref_count[block_J.data] = 1;
    int dim[4];
    int cnt = 0;
    for (int ii = 0; ii < A.shape.size(); ii++) {
        if (ii == dim0 || ii == dim1) continue;
        dim[cnt++] = ii;
    }
    int scan[6];
    scan[5] = 1;
    for (int ii = 4; ii >= 0; ii--) {
        scan[ii] = scan[ii+1] * shape[ii+1];
    }
    int scan_add[6] = {0};
    for (int ii = 1; ii < 6; ii++) {
        scan_add[ii] = scan_add[ii-1] + shape[ii-1];
    }

    for (int ii = 0; ii < shape[dim0]; ii++) {
        int idx_ii = ii * scan[dim0];
        int block_idx_ii = ii * shape[dim1];
        for (int jj = 0; jj < shape[dim1]; jj++) {
            int idx_jj = jj * scan[dim1] + idx_ii;
            int block_idx = block_idx_ii + jj;
            for (int kk = 0; kk < shape[dim[0]]; kk++) {
                int idx_kk = kk * scan[dim[0]] + idx_jj;
                for (int ll = 0; ll < shape[dim[1]]; ll++) {
                    int idx_ll = ll * scan[dim[1]] + idx_kk;
                    for (int uu = 0; uu < shape[dim[2]]; uu++) {
                        int idx_uu = uu * scan[dim[2]] + idx_ll;
                        for (int tt = 0; tt < shape[dim[3]]; tt++) {
                            block_J.data[block_idx] += 
                                A.data[idx_uu + tt]* U.data[scan_add[dim[0]]+kk] * U.data[scan_add[dim[1]]+ll]
                                * U.data[scan_add[dim[2]]+uu] * U.data[scan_add[dim[3]]+tt];
                        }
                    }
                }
            }
        }
    }

    return block_J;
}

void svd_solve(Tensor &J, Tensor &eigvec, double &eig) {
    lapack_int n = J.shape[0];
    lapack_int lda = n;
    double w[n];
    double *tmp_j = (double*)std::malloc(sizeof(double) * J.size);
    memcpy(tmp_j, J.data, sizeof(double) * J.size);

    // mkl omp
    auto info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, tmp_j, lda, w);
    if (info != 0) {
        std::cout << "Error syev @" << __LINE__ << std::endl;
    }
    eig = w[n-1];
    int idx = n - 1;
    if (abs(w[0] > abs(w[n-1]))) {
        eig = w[0];
        idx = 0;
    }
    for (int ii = 0; ii < n; ii++) {
        eigvec.data[ii] = tmp_j[ii*n+idx];
    }
    std::free(tmp_j);
    return ;
}

void fill_J_with_block(Tensor &J, vint shapeA, int x, int y, Tensor &block) {
    int n_J = J.shape[0];
    int x_begin = 0;
    int y_begin = 0;
    int n_x = shapeA[x];
    int n_y = shapeA[y];
    for (int i = 0; i < x; i++)
        x_begin += shapeA[i];
    for (int i = 0; i < y; i++)
        y_begin += shapeA[i];
    
    for (int i = 0; i < n_x; i++) {
        std::memcpy(J.data + (i+x_begin)*n_J + y_begin, block.data + i*n_J, sizeof(double) * n_y);
    }
    return ;
}

double cal_res(Tensor& J, Tensor &X, double lambda) {
    Tensor w_inter;
    w_inter.size = X.size;
    w_inter.ndim = 1;
    w_inter.shape.push_back(X.shape[0]);
    w_inter.data = (double*)std::malloc(sizeof(double)*X.size);
    std::memset(w_inter.data, 0, sizeof(double) * X.size);
    for (int ii = 0; ii < J.shape[0]; ii++) {
        int idx = ii * J.shape[1];
        for (int jj = 0; jj < J.shape[1]; jj++) {
            w_inter.data[ii] += J.data[idx+jj] * X.data[jj];
        }
    }
    double rho;
    for (int ii = 0; ii < X.size; ii++) {
        rho += w_inter.data[ii] * X.data[ii];
    }
    for (int ii = 0; ii < X.size; ii++) {
        w_inter.data[ii] -= rho * X.data[ii];
    }
    return fnorm(w_inter)/(fnorm(J)+abs(lambda));
}

double scf(Tensor &A, std::vector<Tensor> &U, double tol, int max_iter) {
    int n = A.ndim;
    vint shape = A.shape;
    double AF = fnorm(A);
    int iter = 0;
    int n_j = 0;
    int scan_nj[7];
    scan_nj[0] = 0;
    for (int ii = 0; ii < n; ii++) {
        n_j += shape[ii];
        scan_nj[ii+1] = n_j;
    }

    Tensor J;
    Tensor X;
    J.size = n_j * n_j;
    J.ndim = 2;
    J.shape={n_j, n_j};
    J.data = (double *)std::malloc(sizeof(double) * J.size);
    std::memset(J.data, 0, sizeof(double) * J.size);
    ref_count[J.data] =1;
    X.size = n_j;
    X.ndim = 1;
    X.shape = {X.size};
    X.data = (double *)std::malloc(sizeof(double) * X.size);
    for (int ii = 0; ii < n; ii++) {
        std::memcpy(X.data + sizeof(double) * scan_nj[ii],
                    U[ii].data, shape[ii] * sizeof(double));
    }
    ref_count[X.data] =1;
    auto lambda = cal_lambda(J, X);

    while (iter < max_iter) {
        // update J
        // TODO : omp
        for (int ii = 1; ii < n - 1; ii++) {
            for (int jj = ii + 1; jj < n; jj++) {
                auto block_J = ttvc_except_dim(A, X, ii, jj);
                Nmul(block_J, n - 1);
                fill_J_with_block(J, shape, ii, jj, block_J);
            }
        }
        Nmul(X, fnorm(X));
        
        auto res = cal_res(J, X, lambda);
        std::cout << iter << "-th scf iteration: lambda is " << lambda << ", residual is" << res << std::endl;
        if (res < tol) {
            break;
        }

        // update X and lambda
        svd_solve(J, X, lambda);
        iter++;
    }
    // assign U
    for (int ii = 0; ii < n; ii++) {
        std::memcpy(U[ii].data, X.data + sizeof(double) * scan_nj[ii],
                    shape[ii] * sizeof(double));
    }
    return lambda;
    
}


int main(int argc, char **argv) {
    vint shapeA = {100,10,10,10,10,10};
    int ndim = shapeA.size();
    Tensor A;
    for (int ii = 0; ii < ndim; ii++) {
        A.shape.push_back(shapeA[ii]);
    }
    A.ndim = A.shape.size();
    A.size = 1;
    for(int ii = 0; ii < ndim; ii++)
        A.size *= A.shape[ii];
    A.data = (double *)std::malloc(sizeof(double) * A.size);
    ref_count[A.data] = 1;

    for(int ii = 0; ii < A.shape[0]; ii++) {
        for(int jj = 0; jj < A.shape[1]; jj++)
            for(int kk = 0; kk < A.shape[2]; kk++)
                for(int ll = 0; ll < A.shape[3];ll++)
                A.data[ii*A.shape[1]*A.shape[2]*A.shape[3]+jj*A.shape[2]*A.shape[3]+kk*A.shape[3]+ll] 
                    = std::sin(ii+jj+kk+ll);
    }
    
    std::vector<Tensor> U;
    for(int ii = 0; ii < ndim; ii++) {
        Tensor u;
        u.ndim = 1;
        u.size = shapeA[ii];
        u.shape.push_back(shapeA[ii]);
        u.data = (double*)std::malloc(sizeof(double) * u.size);
        ref_count[u.data] =1;
        for(int jj = 0; jj < u.size; jj++)
            u.data[jj] = randn();
        double a = fnorm(u);
        for(int jj = 0; jj < u.size; jj++)
            u.data[jj] /= a;
        U.push_back(u);
    }
    auto tt = tnow();
    double lamb = scf(A, U, 1e-12, 100);
    pti(tt);
}
