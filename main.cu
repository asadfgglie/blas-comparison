#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <cuda_runtime.h>

extern "C" {
#include "utils.h"
}

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

#define CUBLAS_CHECK(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error: %d\n", err); \
        exit(EXIT_FAILURE); \
    }

/**
 * use Steepest Descent to solve Ax = b
 * @param matrix n*n SPD matrix A with packed form
 * @param n dim
 * @param b nonhomogenous term for Ax = b
 * @param x sol.
 * @param tolerance tolerance range
 */
long long d_steepest_descent_cuda(double const *matrix, long long const n, double const *b, double *x, double const tolerance) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    double *d_matrix, *d_b, *d_x, *d_r, *d_y;
    size_t const matrix_size = sizeof(double) * n * (n + 1) / 2;
    size_t const vec_size = sizeof(double) * n;

    CUDA_CHECK(cudaMalloc((void**)&d_matrix, matrix_size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, vec_size));
    CUDA_CHECK(cudaMalloc((void**)&d_x, vec_size));
    CUDA_CHECK(cudaMalloc((void**)&d_r, vec_size));
    CUDA_CHECK(cudaMalloc((void**)&d_y, vec_size));

    // Upload data to device
    CUDA_CHECK(cudaMemcpy(d_matrix, matrix, matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, vec_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, vec_size));

    // r <- b
    CUBLAS_CHECK(cublasDcopy(handle, n, d_b, 1, d_r, 1));

    long long iter = 0;
    double e = INFINITY;

    double alpha_num, alpha_den, alpha;
    double neg_one = -1.0;
    double one = 1.0;
    double zero = 0.0;

    // Note: C RowMajor Upper Packed is exactly identical to ColumnMajor Lower packed in memory!
    // So we safely use CUBLAS_FILL_MODE_LOWER
    while (e > tolerance) {
        CUBLAS_CHECK(cublasDspmv(handle, CUBLAS_FILL_MODE_LOWER, n, &one, d_matrix, d_r, 1, &zero, d_y, 1));
        CUBLAS_CHECK(cublasDdot(handle, n, d_r, 1, d_r, 1, &alpha_num));
        CUBLAS_CHECK(cublasDdot(handle, n, d_r, 1, d_y, 1, &alpha_den));
        alpha = alpha_num / alpha_den;
        // alpha <- r^T r / r^T y

        CUBLAS_CHECK(cublasDaxpy(handle, n, &alpha, d_r, 1, d_x, 1));
        // x <- x + alpha * r

        CUBLAS_CHECK(cublasDspmv(handle, CUBLAS_FILL_MODE_LOWER, n, &neg_one, d_matrix, d_x, 1, &zero, d_y, 1));
        CUBLAS_CHECK(cublasDaxpy(handle, n, &one, d_b, 1, d_y, 1));
        CUBLAS_CHECK(cublasDcopy(handle, n, d_y, 1, d_r, 1));
        // r <- -Ax + b

        CUBLAS_CHECK(cublasDnrm2(handle, n, d_r, 1, &e));
        // e <- norm(r)

        iter++;
    }

    CUDA_CHECK(cudaMemcpy(x, d_x, vec_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_matrix));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_y));
    CUBLAS_CHECK(cublasDestroy(handle));

    return iter;
}

int main(int const argc, char **argv) {
    srand48(time(nullptr));

    auto *filename = "./result.txt";
    long long n = 10;
    if (argc >= 2) {
        n = atoi(argv[1]);
    }
    if (argc >= 3) {
        filename = argv[2];
    }
    int verbose = 0;
    if (argc >= 4) {
        verbose = atoi(argv[3]);
    }

    double const start_gen_SPD = timer();
    auto *A = static_cast<double *>(malloc(sizeof(double) * n * (n + 1) / 2));
    d_gen_SPD(A, n, 10);
    double const end_gen_SPD = timer();
    if (verbose) {
        printf("SPD A matrix:\n");
        d_print_SPD_matrix(A, n);
    }
    printf("\ngen SPD timer: %.5fs\n", end_gen_SPD - start_gen_SPD);

    double const start_gen_vec = timer();
    auto *b = static_cast<double *>(malloc(sizeof(double) * n));
    d_gen_vec(b, n);
    double const end_gen_vec = timer();
    if (verbose) {
        printf("\nb vector:\n");
        d_print_vec(b, n);
    }
    printf("\ngen b vector timer: %.5fs\n", end_gen_vec - start_gen_vec);

    auto *x = static_cast<double *>(malloc(sizeof(double) * n));

    double const start = timer();
    long long const iter = d_steepest_descent_cuda(A, n, b, x, 1e-10);
    double const end = timer();
    if (verbose) {
        printf("\nsol. x vector:\n");
        d_print_vec(x, n);
    }
    printf("\ntimer: %.5fs, iter: %lld\n", end - start, iter);

    d_save_SPD_system(A, n, b, x, filename);
    printf("\nprecision linear system is save into `%s`.\n", filename);

    free(A);
    free(b);
    free(x);
    return 0;
}