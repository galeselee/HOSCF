Best rank-one approximation is one of the most fundamental tasks in tensor computation. In order to fully exploit modern multi-core parallel computers, we build a bridge between the rank-one approximation of tensors and the eigenvector-dependent nonlinear eigenvalue problem (NEPv), and then develop an efficient decoupling algorithm, namely the higher-order self-consistent field (HOSCF) algorithm. We implement the HOSCF algorithm in C++ language with OpenMP multi-threading and MPI multi-processes, and the involved linear algebra operations such as eigensolver available from the Intel MKL library.

## ENV
There are two ways for env, spack and oneapi.
### [spack](https://spack.readthedocs.io/en/latest/getting_started.html#)
```
gcc-12
openmpi-4.1.4
mkl-2020
gomp
```

### [Oneapi](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
1. install oneAPI Base and hpc Toolkit
2. source setvars.sh

## Compile
```
# SCF
1. mkdir build
2. cmake ..
3. make -j 9

# ALS
1. cd als
2. mkdir build
3. cmake ..
4. make -j 9
```

## SCF strong scalability
| Threads    | Time/ms | scalability(based on num_thread=1)|
| -------- | ------- | --------------|
| 1 | 14305 | - |
| 2 | 7401.1 |0.966|
| 4 | 3735.6 |0.957|
| 8 | 1925.9 |0.928|
| 16 | 1106.5 | 0.808|
| 32(with two mpi processes) | 578 |0.773 |
