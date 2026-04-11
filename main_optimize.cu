#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cublas_v2.h>
#include <curand.h>
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

#define CURAND_CHECK(err) \
    if (err != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "cuRAND Error: %d\n", err); \
        exit(EXIT_FAILURE); \
    }

__global__ void shift_matrix_kernel(double *mat, long long const size) {
    long long const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        mat[idx] -= 0.5;
    }
}

__global__ void pack_and_shift_diag_kernel(double *A_packed, const double *result, long long const n, double const e) {
    long long const row = blockIdx.y * blockDim.y + threadIdx.y;
    long long const col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n && row <= col) {
        long long const packed_idx = row * n - row * (row - 1) / 2 + (col - row);
        double val = result[row * n + col];
        if (row == col) {
            val += e;
        }
        A_packed[packed_idx] = val;
    }
}

/**
 * Generate a row major SPD n*n matrix with packed form
 * @param cublasH cublas handler
 * @param curandG cuda random generator
 * @param d_A data desc
 * @param n matrix dim
 * @param k condition approximation, must greater than 1
 */
void d_gen_SPD_gpu(cublasHandle_t const cublasH, curandGenerator_t const curandG, double *d_A, long long const n, double k) {
    double *d_matrix, *d_result;
    CUDA_CHECK(cudaMalloc((void**)&d_matrix, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_result, n * n * sizeof(double)));

    // Generate uniform (0, 1]
    CURAND_CHECK(curandGenerateUniformDouble(curandG, d_matrix, n * n));

    // Shift by -0.5
    long long const total_elements = n * n;
    long long constexpr threads = 256;
    long long const blocks = (total_elements + threads - 1) / threads;
    shift_matrix_kernel<<<blocks, threads>>>(d_matrix, total_elements);
    CUDA_CHECK(cudaGetLastError());

    // GEMM: d_result = d_matrix^T * d_matrix
    // Since flat arrays are column-major for cuBLAS, passing CUBLAS_OP_T for A and CUBLAS_OP_N for B
    // gives R_col = A_col^T * A_col, which is identically R_row = A_row * A_row^T in memory!
    double constexpr alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &alpha, d_matrix, n, d_matrix, n, &beta, d_result, n));

    k = k > 1 ? k : 1 + 1e-10;
    double const e = static_cast<double>(n) / (3 * (k - 1));

    // Pack and add diagonal
    dim3 blockDim(16, 16);
    dim3 gridDim((n + 15) / 16, (n + 15) / 16);
    pack_and_shift_diag_kernel<<<gridDim, blockDim>>>(d_A, d_result, n, e);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_matrix));
    CUDA_CHECK(cudaFree(d_result));
}

void d_gen_vec_gpu(curandGenerator_t curandG, double *d_vec, long long const n) {
    CURAND_CHECK(curandGenerateUniformDouble(curandG, d_vec, n));
}

/**
 * use Steepest Descent to solve Ax = b
 * @param handle cublas handler
 * @param d_matrix n*n SPD matrix A with packed form
 * @param n dim
 * @param d_b nonhomogenous term for Ax = b
 * @param d_x sol.
 * @param tolerance tolerance range
 */
long long d_steepest_descent_cuda_opt(cublasHandle_t const handle, double const *d_matrix, long long const n, double const *d_b, double *d_x, double const tolerance) {
    double *d_r, *d_y;
    size_t const vec_size = sizeof(double) * n;

    CUDA_CHECK(cudaMalloc((void**)&d_r, vec_size));
    CUDA_CHECK(cudaMalloc((void**)&d_y, vec_size));

    CUDA_CHECK(cudaMemset(d_x, 0, vec_size));
    // x <- 0

    CUBLAS_CHECK(cublasDcopy(handle, n, d_b, 1, d_r, 1));
    // r <- b

    long long iter = 0;
    double e = INFINITY;

    double alpha_num, alpha_den, alpha;
    double constexpr neg_one = -1.0;
    double constexpr one = 1.0;
    double constexpr zero = 0.0;

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

    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_y));

    return iter;
}

int main(int const argc, char **argv) {
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

    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate(&cublasH));

    curandGenerator_t curandG;
    CURAND_CHECK(curandCreateGenerator(&curandG, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curandG, time(nullptr)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    size_t const packed_size = sizeof(double) * n * (n + 1) / 2;
    size_t const vec_size = sizeof(double) * n;

    double *d_A, *d_b, *d_x;
    CUDA_CHECK(cudaMalloc((void**)&d_A, packed_size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, vec_size));
    CUDA_CHECK(cudaMalloc((void**)&d_x, vec_size));

    cudaEventRecord(start);
    d_gen_SPD_gpu(cublasH, curandG, d_A, n, 10.0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("\ngen SPD timer (GPU): %.17fs\n", static_cast<double>(ms) / 1000.0);

    cudaEventRecord(start);
    d_gen_vec_gpu(curandG, d_b, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("\ngen b vector timer (GPU): %.17fs\n", static_cast<double>(ms) / 1000.0);

    cudaEventRecord(start);
    long long const iter = d_steepest_descent_cuda_opt(cublasH, d_A, n, d_b, d_x, 1e-10);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("\nsteepest_descent timer (GPU): %.17fs, iter: %lld\n", static_cast<double>(ms) / 1000.0, iter);

    // download and Save
    auto *h_A = static_cast<double *>(malloc(packed_size));
    auto *h_b = static_cast<double *>(malloc(vec_size));
    auto *h_x = static_cast<double *>(malloc(vec_size));

    CUDA_CHECK(cudaMemcpy(h_A, d_A, packed_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b, d_b, vec_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_x, d_x, vec_size, cudaMemcpyDeviceToHost));

    if (verbose) {
        printf("\nSPD A matrix:\n");
        d_print_SPD_matrix(h_A, n);
        printf("\nb vector:\n");
        d_print_vec(h_b, n);
        printf("\nsol. x vector:\n");
        d_print_vec(h_x, n);
    }

    d_save_SPD_system(h_A, n, h_b, h_x, filename);
    printf("\nprecision linear system is save into `%s`.\n", filename);

    free(h_A);
    free(h_b);
    free(h_x);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    curandDestroyGenerator(curandG);
    cublasDestroy(cublasH);

    return 0;
}
