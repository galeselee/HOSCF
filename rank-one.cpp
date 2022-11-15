
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

Tensor permute(Tensor &A, vint perm) {
    Tensor ret;
    vint shape = A.shape;
    ret.size = A.size;
    ret.ndim = A.ndim;
    for(int ii = 0; ii < ret.ndim; ii++) 
        ret.shape.push_back(A.shape[perm[ii]]);
    ret.data = (double *)std::malloc(sizeof(double) * ret.size);
    ref_count[ret.data] ++;

    int size = 1;
    int ndim = ret.ndim;
    vint shape_new(ndim);
    vint stride_new(ndim);
    for (int i = 0; i < ndim; i++) {
        shape_new[i] = shape[perm[i]];
        size *= shape[i];
    }
    stride_new[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) stride_new[i] = stride_new[i + 1] * shape_new[i + 1];
    vint stride_new_perm(ndim);
    for (int i = 0; i < ndim; i++) stride_new_perm[perm[i]] = stride_new[i];

    vint idx(ndim);
    int index_new = 0;
    for (int i = 0; i < size; i++) {
        ret.data[index_new] = A.data[i];
        for (int d = ndim - 1; d >= 0; d--) {
            int snpd = stride_new_perm[d];
            int sd = shape[d];
            idx[d]++;
            index_new += snpd;
            if (idx[d] != sd) {
                break;
            } else {
                idx[d] = 0;
                index_new -= snpd * sd;
            }
        }
    }
    return ret;
}

Tensor kr(Tensor &A, Tensor &B) {
    Tensor ret;
    ret.ndim = A.ndim;
    if(ret.ndim == 2) {
        ret.shape = {A.shape[0] * B.shape[0], A.shape[1]};
        ret.size = A.shape[0] * B.shape[0] * A.shape[1];
    }
    if(ret.ndim == 1) {
        ret.shape = {A.shape[0] * B.shape[0]};
        ret.size = A.shape[0] * B.shape[0];
    }
    ret.data = (double *)std::malloc(sizeof(double) * ret.size); 
    ref_count[ret.data] ++;
    if(ret.ndim == 2) {
        int M_A = A.shape[0], N_A = A.shape[1];
        int M_B = B.shape[0], N_B = B.shape[1];
        for (int i = 0; i < M_A * M_B; i++) {
            int ic_s = i * N_A;
            int ia_s = (i / M_B) * N_A;
            int ib_s = (i % M_B) * N_A;
            for (int j = 0; j < N_A; j++) {
                ret.data[ic_s + j] = A.data[ia_s + j] * B.data[ib_s + j];
            }
        }
    }
    if(ret.ndim == 1) {
        int M_A = A.shape[0];
        int M_B = B.shape[0];
        for (int i = 0; i < M_A * M_B; i++) {
            int ic_s = i;
            int ia_s = i / M_B;
            int ib_s = i % M_B;
            ret.data[ic_s] = A.data[ia_s] * B.data[ib_s];
        }
    }
    return ret;
}

Tensor solve(Tensor &A, Tensor &u) {
    Tensor x = u;
    lapack_int m = A.shape[0];
    lapack_int LDA = m;
    lapack_int LDB = 1;
    lapack_int NRHS = 1;
    lapack_int INFO;
    lapack_int *IPIV = nullptr;
    IPIV = (lapack_int *)std::malloc(m * sizeof(lapack_int));
    char TRANS = 'N';
    double *A_ = (double *)std::malloc(sizeof(double) * A.size);
    std::memcpy(A_, A.data, A.size * sizeof(double));
    INFO = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, m, A_, LDA, IPIV);
    if(INFO != 0) printf("Warning INFO != 0, function : solve, order = 1\n");
    INFO = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, TRANS, m, NRHS, A_, LDA, IPIV, x.data, LDB);
    if(INFO != 0) printf("Warning INFO != 0, function : solve, order = 2\n");
    std::free(A_);
    std::free(IPIV);
    return x;
}

Tensor ttv(Tensor &A, Tensor &u, int dim) {
    Tensor A_;
    for(int ii = 0; ii < A.ndim; ii++) {
        if(ii == dim) continue;
        A_.shape.push_back(A.shape[ii]);
    }
    A_.size = A.size / A.shape[dim];
    A_.ndim = A.ndim - 1;
    A_.data = (double *)std::malloc(sizeof(double) * A_.size);
    std::memset(A_.data, 0, sizeof(double) * A_.size);
    ref_count[A_.data] ++;
    int out_dim = 1;
    int in_dim = 1;
    for (int ii = 0; ii < dim; ii++)
        out_dim *= A.shape[ii];
    for (int ii = dim; ii < A.ndim - 1; ii++)
        in_dim *= A.shape[ii+1];
    for (int i = 0; i < out_dim; i++) {
        for (int j = 0; j < A.shape[dim]; j++) {
            for (int k = 0; k < in_dim; k++) {
                 A_.data[i * in_dim + k] += 
                        A.data[i * A.shape[dim] * in_dim + j * in_dim + k] * u.data[j];
            }
        }
    }
    return A_;
}

void grqi_push_vector_into_U(std::vector<Tensor> &U, Tensor &u) {
    int n = U.size();
    int pos = 0;
    for(int ii = 0; ii < n; ii++) {
        int delta = U[ii].size;
        std::memcpy(U[ii].data, u.data + pos, delta * sizeof(double));
        pos += delta;
    }
}

void grqi_push_in_vector_B(Tensor &b, vint shapeA, int xx, Tensor &v) {
    int n_b = b.shape[0];
    int n = shapeA[xx];
    int x_b = 0;
    for(int ii = 0; ii < xx; ii++) x_b += shapeA[ii];
    std::memcpy(b.data + x_b, v.data, n * sizeof(double));
}

void grqi_push_in_matrix_J(Tensor &J, vint shapeA, int x, int y, Tensor &mat) {
    int n_J = J.shape[0];
    int x_begin = 0;
    int y_begin = 0;
    int N = shapeA[x];
    int M = shapeA[y];
    for (int i = 0; i < shapeA.size(); i++) {
        if (i < x) {
            x_begin += shapeA[i];
        }
        if (i < y) {
            y_begin += shapeA[i];
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            int i_now = i + x_begin;
            int j_now = j + y_begin;
            double dat_now = mat.data[i * M + j];
            J.data[i_now * n_J + j_now] = dat_now;
            J.data[j_now * n_J + i_now] = dat_now;
        }
    }
}

void grqi_gen_J_diag(Tensor &J, int N_J, double nlamb) {
    for (int i = 0; i < N_J; i++) {
        J.data[i * N_J + i] = nlamb;
    }
}

std::tuple<Tensor, Tensor>
grqi_generate_J_and_b(Tensor &A, std::vector<Tensor> &U, double lambda) {
    int N = A.ndim;
    vint shape = A.shape;
    int N_J = 0;
    for (auto i : shape) {
        N_J += i;
    }
    Tensor J;
    Tensor b;
    J.size = N_J * N_J;
    J.ndim = 2;
    J.shape={N_J, N_J};
    J.data = (double *)std::malloc(sizeof(double) * J.size);
    std::memset(J.data, 0, sizeof(double) * J.size);
    ref_count[J.data] ++;
    b.size = N_J;
    b.ndim = 1;
    b.shape = {b.size};
    b.data = (double *)std::malloc(sizeof(double) * b.size);
    std::memset(b.data, 0, sizeof(double) * b.size);
    ref_count[b.data] ++;
    grqi_gen_J_diag(J, N_J, -lambda);

    Tensor K[N];
    K[N-1] = U[N-1];
    for (int i = N - 2; i >= 2; i--) {
        K[i] = kr(U[i], K[i + 1]);
    }
    Tensor B;
    B.copy(A);
    for (int i = 0; i < N - 1; i++) {
        Tensor C;
        C.copy(B);
        for (int j = i + 1; j < N; j++) {
            if (j != N - 1) {
                vint C_shape = C.shape;
                C.shape=vint{C_shape[0], C_shape[1], K[j + 1].size};
                Tensor tempc = ttv(C, K[j + 1], 2);
                grqi_push_in_matrix_J(J, shape, i, j, tempc);
                C.shape = C_shape;
                C = ttv(C, U[j], 1);
            } else {
                grqi_push_in_matrix_J(J, shape, i, j, C);
                C = ttv(C, U[j], 1);
                grqi_push_in_vector_B(b, shape, i, C);
            }
        } 
        B = ttv(B, U[i], 0);
    }
    grqi_push_in_vector_B(b, shape, N - 1, B);
    return std::make_tuple(J, b);
}

std::tuple<Tensor, Tensor>
grqi_raw_generate_J_and_b(Tensor &A, std::vector<Tensor> &U, double lambda) {
    int N = A.ndim;
    vint shape = A.shape;
    int N_J = 0;
    for (auto i : shape) {
        N_J += i;
    }
    Tensor J;
    Tensor b;
    J.size = N_J * N_J;
    J.ndim = 2;
    J.shape={N_J, N_J};
    J.data = (double *)std::malloc(sizeof(double) * J.size);
    std::memset(J.data, 0, sizeof(double) * J.size);
    ref_count[J.data] ++;
    b.size = N_J;
    b.ndim = 1;
    b.shape = {b.size};
    b.data = (double *)std::malloc(sizeof(double) * b.size);
    std::memset(b.data, 0, sizeof(double) * b.size);
    ref_count[b.data] ++;
    grqi_gen_J_diag(J, N_J, -lambda);

    Tensor K[N];
    K[N-1] = U[N-1];
    for (int i = N - 2; i >= 2; i--) {
        K[i] = kr(U[i], K[i + 1]);
    }
    for (int i = 0; i < N - 1; i++) {
        for (int j = i + 1; j < N; j++) {
            Tensor C;
            C.copy(A);
            int skip = 0;
            for(int kk = 0; kk < N; kk++) {
                if(kk == i || kk == j) {
                    skip++;
                    continue;
                }
                C = ttv(C, U[kk], skip);
            }
            grqi_push_in_matrix_J(J, shape, i, j, C);
        } 
    }
    for(int ii = 0; ii < N; ii++) {
        Tensor C;
        C.copy(A);
        int skip = 0;
        for(int jj = 0; jj < N; jj++) {
            if(jj==ii) {
                skip ++;
                continue;
            }
            C = ttv(C, U[jj], skip);
        }
        grqi_push_in_vector_B(b, shape, ii, C);
    }
    return std::make_tuple(J, b);
}

double grqi(Tensor &A, std::vector<Tensor> &U, double tol, int max_iter) {
    int n = A.ndim;
    vint shape = A.shape;
    double AF = fnorm(A);
    vint perm(n);
    for (int i = 0; i < n; i++) {
        perm[i] = i;
    }
    std::sort(perm.begin(), perm.end(),
              [&](const int &i, const int &j) -> bool {
                  return shape[i] > shape[j];
              });
              
    auto A_p = permute(A, perm);
    std::vector<Tensor> U_p(n);
    for (int i = 0; i < n; i++) {
        U_p[i].copy(U[perm[i]]);
    }
    Tensor L;
    L.copy(A_p);
    for(int ii = 0; ii < n; ii++) {
        L = ttv(L, U_p[ii], 0);
    }
    double lambda = L.data[0];
    double residual = 0;
    double residual_last = std::sqrt(1 - lambda * lambda /(AF*AF));
    
    int iter = 0;
    while (std::abs(residual_last - residual) > tol && iter < max_iter) {
        auto [J, b] = grqi_generate_J_and_b(A_p, U_p, lambda);
        Tensor u = solve(J, b);
        Nmul(u, n-2);
        grqi_push_vector_into_U(U_p, u);
        for(int ii = 0; ii < n; ii++) Nmul(U_p[ii], 1/fnorm(U_p[ii]));
        L.copy(A_p);
        for(int ii = 0; ii < n; ii++) L = ttv(L, U_p[ii], 0);
        lambda = L.data[0];
        residual_last = residual;
        residual = std::sqrt(1 - (lambda * lambda) / (AF * AF));
        iter++;
        std::cout<<"Iteration " << std::to_string(iter) <<
             ": lambda = " << std::to_string(lambda) <<
             "; residual = " << std::to_string(residual) << "; error_delta = " <<
             std::to_string(std::abs(residual_last - residual)) <<std::endl;
    }
    for(int ii = 0; ii < n; ii++) U[ii].copy(U_p[ii]);
    return lambda;
}

double grqi_raw(Tensor &A, std::vector<Tensor> &U, double tol, int max_iter) {
    int n = A.ndim;
    vint shape = A.shape;
    double AF = fnorm(A);
    vint perm(n);
    for (int i = 0; i < n; i++) {
        perm[i] = i;
    }
    std::sort(perm.begin(), perm.end(),
              [&](const int &i, const int &j) -> bool {
                  return shape[i] > shape[j];
              });
              
    auto A_p = permute(A, perm);
    std::vector<Tensor> U_p(n);
    for (int i = 0; i < n; i++) {
        U_p[i].copy(U[perm[i]]);
    }
    Tensor L;
    L.copy(A_p);
    for(int ii = 0; ii < n; ii++) {
        L = ttv(L, U_p[ii], 0);
    }
    double lambda = L.data[0];
    double residual = 0;
    double residual_last = std::sqrt(1 - lambda * lambda /(AF*AF));
    
    int iter = 0;
    while (std::abs(residual_last - residual) > tol && iter < max_iter) {
        auto [J, b] = grqi_raw_generate_J_and_b(A_p, U_p, lambda);
        Tensor u = solve(J, b);
        Nmul(u, n-2);
        grqi_push_vector_into_U(U_p, u);
        for(int ii = 0; ii < n; ii++) Nmul(U_p[ii], 1/fnorm(U_p[ii]));
        L.copy(A_p);
        for(int ii = 0; ii < n; ii++) L = ttv(L, U_p[ii], 0);
        lambda = L.data[0];
        residual_last = residual;
        residual = std::sqrt(1 - (lambda * lambda) / (AF * AF));
        iter++;
        std::cout<<"Iteration " << std::to_string(iter) <<
             ": lambda = " << std::to_string(lambda) <<
             "; residual = " << std::to_string(residual) << "; error_delta = " <<
             std::to_string(std::abs(residual_last - residual)) <<std::endl;
    }
    for(int ii = 0; ii < n; ii++) U[ii].copy(U_p[ii]);
    return lambda;
}

double hopm(Tensor &A, std::vector<Tensor> &U, double tol, int max_iter) {
    int N = A.ndim;
    vint shape = A.shape;
    double AF = fnorm(A);
    double residual = 0;
    double residual_last = 2 * tol;
    int iter = 0;
    Tensor L;
    L.copy(A);
    for(int ii = 0; ii < N; ii++) {
        L = ttv(L, U[ii], 0);
    }
    double lambda = L.data[0];
    while (std::abs(residual_last - residual) > tol && iter < max_iter) {
        for (int i = 0; i < N; i++) {
            Tensor C;
            C.copy(A);
            for (int k = 0; k < i; k++)
                C = ttv(C, U[k], 0);
            for (int k = i + 1; k < N; k++)
                C = ttv(C, U[k], 1);
            std::memcpy(U[i].data, C.data, C.size * sizeof(double));
            Nmul(U[i], 1 / fnorm(U[i]));
        }
        L.copy(A);
        for(int ii = 0; ii < N; ii++) {
            L = ttv(L, U[ii], 0);
        }
        lambda = L.data[0];
        residual_last = residual;
        residual = std::sqrt(1 - (lambda * lambda) / (AF * AF));
        iter++;
        std::cout<<"Iteration " << std::to_string(iter) <<
        ": lambda = " << std::to_string(lambda) <<
        "; residual = " << std::to_string(residual) << "; error_delta = " <<
        std::to_string(std::abs(residual_last - residual)) <<std::endl;
    }
    return lambda;
}

int main(int argc, char **argv) {
    vint shapeA = {100,100,100,100};
    int ndim = shapeA.size();
    Tensor A;
    A.shape.push_back(shapeA[0]);
    A.shape.push_back(shapeA[1]);
    A.shape.push_back(shapeA[2]);
    A.shape.push_back(shapeA[3]);
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
        ref_count[u.data] ++;
        for(int jj = 0; jj < u.size; jj++)
            u.data[jj] = randn();
        double a = fnorm(u);
        for(int jj = 0; jj < u.size; jj++)
            u.data[jj] /= a;
        U.push_back(u);
    }
    auto tt = tnow();
    // double lamb = grqi(A, U, 1e-12, 100);
    double lamb = grqi_raw(A, U, 1e-12, 100);
    // double lamb = hopm(A, U, 1e-12, 100);
    pti(tt);
}
