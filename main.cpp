
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
        for(int ii = 0; ii < b.ndim; ii++) {
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
        for(int ii = 0; ii < b.ndim; ii++) {
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
    for (int i = 0; i < n; i++) {
        norm += A.data[i] * A.data[i];
    }
    return std::sqrt(norm);
}

double fnorm_ptr(double *ptr, int size) {
    double norm = 0;
    for (int i = 0; i < size; i++) {
        norm += ptr[i] * ptr[i];
    }
    return std::sqrt(norm);
}

void Nmul(Tensor &A, double num) {
    for(int ii = 0; ii < A.size; ii++) {
        A.data[ii] *= num;
    }
}

void Nmul_ptr(double *ptr, double num, int size) {
    for(int ii = 0; ii < size; ii++) {
        ptr[ii] *= num;
    }
}

double cal_lambda(Tensor &A, Tensor &U) {
    vint shape = A.shape;
    double lambda = 0;
    int scan[6];
    scan[5] = 1;
    for (int ii = 4; ii >= 0; ii--) {
        scan[ii] = scan[ii+1] * shape[ii+1];
    }
    int scan_add[6];
    scan_add[0] = 0;
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
                            lambda += A.data[idx_uu + tt] * U.data[scan_add[0]+tt] * U.data[scan_add[1]+uu]
                                      * U.data[scan_add[2]+ll] * U.data[scan_add[3]+kk]
                                      * U.data[scan_add[4]+jj] * U.data[scan_add[5]+ii];
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
    int ndim = A.ndim;
    int a_dim0 = ndim-1-dim0;
    int a_dim1 = ndim-1-dim1;
    Tensor block_J;
    block_J.size = A.shape[a_dim0] * A.shape[a_dim1];
    block_J.ndim = 2;
    block_J.shape = {A.shape[a_dim0] * A.shape[a_dim1]};
    block_J.data = (double*)std::malloc(sizeof(double) * block_J.size);
    std::memset(block_J.data, 0, sizeof(double) * block_J.size);
    ref_count[block_J.data] = 1;
    int dim[4];
    int cnt = 0;
    for (int ii = 0; ii < A.ndim; ii++) {
        if (ii == a_dim0 || ii == a_dim1) continue;
        dim[cnt++] = ii;
    }
    int scan[6];
    scan[5] = 1;
    for (int ii = 4; ii >= 0; ii--) {
        scan[ii] = scan[ii+1] * shape[ii+1];
    }
    int scan_add[6];
    scan_add[0]=0;
    for (int ii = 1; ii < 6; ii++) {
        scan_add[ii] = scan_add[ii-1] + shape[ndim-ii];
    }


    for (int ii = 0; ii < shape[a_dim0]; ii++) {
        int idx_ii = ii * scan[a_dim0];
        int block_idx_ii = ii * shape[a_dim1];
        for (int jj = 0; jj < shape[a_dim1]; jj++) {
            int idx_jj = jj * scan[a_dim1] + idx_ii;
            int block_idx = block_idx_ii + jj;
            for (int kk = 0; kk < shape[dim[0]]; kk++) {
                int idx_kk = kk * scan[dim[0]] + idx_jj;
                for (int ll = 0; ll < shape[dim[1]]; ll++) {
                    int idx_ll = ll * scan[dim[1]] + idx_kk;
                    for (int uu = 0; uu < shape[dim[2]]; uu++) {
                        int idx_uu = uu * scan[dim[2]] + idx_ll;
                        for (int tt = 0; tt < shape[dim[3]]; tt++) {
                            block_J.data[block_idx] += 
                                A.data[idx_uu + tt*scan[dim[3]]] * U.data[scan_add[ndim-1-dim[3]]+tt] * U.data[scan_add[ndim-1-dim[2]]+uu]
                                * U.data[scan_add[ndim-1-dim[1]]+ll] * U.data[scan_add[ndim-1-dim[0]]+kk];
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

    // mkl omp
    auto info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, J.data, lda, w);
    if (info != 0) {
        std::cout << "Error syev @" << __LINE__ << std::endl;
    }
    eig = w[n-1];
    int idx = n - 1;
    if (std::abs(w[0]) > std::abs(w[n-1])) {
        eig = w[0];
        idx = 0;
    }
    for (int ii = 0; ii < n; ii++) {
        eigvec.data[ii] = J.data[ii*n+idx];
    }
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
        std::memcpy(J.data + (i+x_begin)*n_J + y_begin, block.data + i*n_x, sizeof(double) * n_y);
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
    ref_count[w_inter.data] = 1;
    for (int ii = 0; ii < J.shape[0]; ii++) {
        int idx = ii * J.shape[1];
        for (int jj = ii+1; jj < J.shape[1]; jj++) {
            w_inter.data[ii] += J.data[idx+jj] * X.data[jj];
        }
    }
    for (int ii = 0; ii < J.shape[0]; ii++) {
        for (int jj = 0; jj < ii; jj++) {
            w_inter.data[ii] += X.data[jj] * J.data[jj*J.shape[0]+ii];
        }
    }
// X.T J X
// (w_inter)- rho * w
    double rho;
	// w_inter inner w
    for (int ii = 0; ii < X.size; ii++) {
        rho += w_inter.data[ii] * X.data[ii]; // X -> w
    }
    for (int ii = 0; ii < X.size; ii++) {
        w_inter.data[ii] -= rho * X.data[ii];
    }

    auto res = fnorm(w_inter)/(fnorm(J)*std::sqrt(2)+std::abs(lambda));
    return res;
}

void scf(Tensor &A, std::vector<Tensor> &U, double tol, int max_iter) {
    int n = A.ndim;
    vint shape = A.shape;
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
    ref_count[J.data] =1;
    X.size = n_j;
    X.ndim = 1;
    X.shape = {X.size};
    X.data = (double *)std::malloc(sizeof(double) * X.size);

    // {n_j} X concat
    for (int ii = 0; ii < n; ii++) {
        std::memcpy(X.data + scan_nj[ii],
                    U[ii].data, shape[ii] * sizeof(double));
    }
    ref_count[X.data] =1;

    double lambda = cal_lambda(A, X);;

    while (iter < max_iter) {
        std::memset(J.data, 0, sizeof(double) * J.size);
        // TODO : omp
        for (int ii = 0; ii < n-1; ii++) {
            for (int jj = ii+1; jj < n; jj++) {
                auto block_J = ttvc_except_dim(A, X, ii, jj);
                Nmul(block_J, 1/((double)n-1)); // block / 5

                fill_J_with_block(J, shape, ii, jj, block_J);
            }
        }
        
        Nmul(X, 1/ fnorm(X));
        auto res = cal_res(J, X, lambda);
        std::cout << iter << "-th scf iteration: lambda is " << lambda << ", residual is " << res << std::endl;
        if (res < tol) {
            break;
        }

        // update X and lambda
        svd_solve(J, X, lambda);

        for (int ii = 0; ii < n; ii++) {
            Nmul_ptr(X.data+scan_nj[ii], 1/fnorm_ptr(X.data+scan_nj[ii], shape[ii]), shape[ii]);
        }
        iter++;
    }
    // assign U
    for (int ii = n-1; ii >= 0; ii--) {
        std::memcpy(U[ii].data, X.data + scan_nj[ii],
                    shape[ii] * sizeof(double));
    }
    //return lambda;
    return;
}


int main(int argc, char **argv) {
    vint shapeA = {10,10,10,10,10,10}; 
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


    int scan[6];
    scan[5] = 1;
    for (int ii = 4; ii >= 0; ii--) {
        scan[ii] = scan[ii+1] * shapeA[ii+1];
    }
   
    for (int ii = 0; ii < shapeA[0]; ii++) {
        int idx_ii = ii * scan[0];
        for (int jj = 0; jj < shapeA[1]; jj++) {
            int idx_jj = jj * scan[1] + idx_ii;
            for (int kk = 0; kk < shapeA[2]; kk++) {
                int idx_kk = kk * scan[2] + idx_jj;
                for (int ll = 0; ll < shapeA[3]; ll++) {
                    int idx_ll = ll * scan[3] + idx_kk;
                    for (int uu = 0; uu < shapeA[4]; uu++) {
                        int idx_uu = uu * scan[4] + idx_ll;
                        for (int tt = 0; tt < shapeA[5]; tt++) {
                            A.data[idx_uu + tt] = randn();
                        }
                    }
                }
            }
        }
    }
  
    std::vector<Tensor> U;
    for(int ii = 0; ii < ndim; ii++) {
        Tensor u;
        u.ndim = 1;
        u.size = shapeA[ndim-1-ii];
        u.shape.push_back(shapeA[ndim-1-ii]);
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
    scf(A, U, 5.0e-4, 50);
    pti(tt);
}
