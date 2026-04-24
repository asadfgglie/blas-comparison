#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl.h>
#include <omp.h>
#include <math.h>
#include <sys/time.h>
// CSR format structure
typedef struct {
    int m;
    int n;
    int nnz;
    float *values;
    long long *col_indices;
    long long *row_ptr;
} CSRMatrix;
// CSC format structure
typedef struct {
    int m;
    int n;
    int nnz;
    float *values;
    long long *row_indices;
    long long *col_ptr;
} CSCMatrix;
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}
// TODO: Implement your custom CSR matrix multiplication here
void custom_csr_mm(const CSRMatrix *A, const float *B, float *C, int k) {
    // Student to implement
    // C = A * B
}
// TODO: Implement your custom CSC matrix multiplication here
void custom_csc_mm(const CSCMatrix *A, const float *B, float *C, int k) {
    // Student to implement
    // C = A * B
}
void generate_random_csr(CSRMatrix *A, int m, int n, int target_nnz) {
    A->m = m;
    A->n = n;
    A->nnz = target_nnz;
    A->values = (float *)malloc(target_nnz * sizeof(float));
    A->col_indices = (long long *)malloc(target_nnz * sizeof(long long));
    A->row_ptr = (long long *)calloc(m + 1, sizeof(long long));
    // Simple random distribution of nnz across rows
    int const nnz_per_row = target_nnz / m;
    int const remainder = target_nnz % m;
    int current_nnz = 0;
    for (int i = 0; i < m; i++) {
        A->row_ptr[i] = current_nnz;
        int row_nnz = nnz_per_row + (i < remainder ? 1 : 0);
        // To ensure we don't exceed columns, cap row_nnz to n
        if (row_nnz > n) row_nnz = n;
        for (int j = 0; j < row_nnz; j++) {
            A->values[current_nnz] = rand() / (float)RAND_MAX;
            A->col_indices[current_nnz] = (j * (n / row_nnz)) % n; // simplistic distinct columns
            current_nnz++;
        }
    }
    A->row_ptr[m] = current_nnz;
    // Fix actual nnz if capped
    A->nnz = current_nnz;
}
void generate_random_csc(CSCMatrix *A, int m, int n, int target_nnz) {
    A->m = m;
    A->n = n;
    A->nnz = target_nnz;
    A->values = (float *)malloc(target_nnz * sizeof(float));
    A->row_indices = (long long *)malloc(target_nnz * sizeof(long long));
    A->col_ptr = (long long *)calloc(n + 1, sizeof(long long));
    int nnz_per_col = target_nnz / n;
    int remainder = target_nnz % n;
    int current_nnz = 0;
    for (int j = 0; j < n; j++) {
        A->col_ptr[j] = current_nnz;
        int col_nnz = nnz_per_col + (j < remainder ? 1 : 0);
        if (col_nnz > m) col_nnz = m;
        for (int i = 0; i < col_nnz; i++) {
            A->values[current_nnz] = rand() / (float)RAND_MAX;
            A->row_indices[current_nnz] = (i * (m / col_nnz)) % m;
            current_nnz++;
        }
    }
    A->col_ptr[n] = current_nnz;
    A->nnz = current_nnz;
}
void free_csr(CSRMatrix *A) {
    free(A->values);
    free(A->col_indices);
    free(A->row_ptr);
}
void free_csc(CSCMatrix *A) {
    free(A->values);
    free(A->row_indices);
    free(A->col_ptr);
}
void generate_dense(float *mat, int const row, int const col) {
    for (int i = 0; i < row * col; i++) {
        mat[i] = rand() / (float)RAND_MAX;
    }
}
void run_benchmark(int m, int n_A, int k, int use_csr) {
    int target_nnz = 32 * m;
    printf("\nBenchmarking: A(%d x %d), B(%d x %d), target_nnz=%d, format=%s\n", 
           m, n_A, n_A, k, target_nnz, use_csr ? "CSR" : "CSC");
    float *B = malloc(n_A * k * sizeof(float));
    float *C_mkl = malloc(m * k * sizeof(float));
    float *C_custom = malloc(m * k * sizeof(float));
    generate_dense(B, n_A, k);
    double start, end;
    if (use_csr) {
        CSRMatrix A;
        generate_random_csr(&A, m, n_A, target_nnz);
        // MKL CSR execution
        sparse_matrix_t mkl_A;
        mkl_sparse_s_create_csr(&mkl_A, SPARSE_INDEX_BASE_ZERO, A.m, A.n, 
                                A.row_ptr, A.row_ptr + 1, A.col_indices, A.values);
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_optimize(mkl_A);
        start = get_time();
        mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f, mkl_A, descr, 
                        SPARSE_LAYOUT_ROW_MAJOR, B, k, k, 0.0f, C_mkl, k);
        end = get_time();
        printf("MKL CSR Time: %f s\n", end - start);
        mkl_sparse_destroy(mkl_A);
        // Custom CSR execution
        start = get_time();
        custom_csr_mm(&A, B, C_custom, k);
        end = get_time();
        printf("Custom CSR Time: %f s\n", end - start);
        free_csr(&A);
    } else {
        CSCMatrix A;
        generate_random_csc(&A, m, n_A, target_nnz);
        // MKL CSC execution
        sparse_matrix_t mkl_A;
        mkl_sparse_s_create_csc(&mkl_A, SPARSE_INDEX_BASE_ZERO, A.m, A.n, 
                                A.col_ptr, A.col_ptr + 1, A.row_indices, A.values);
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_optimize(mkl_A);
        start = get_time();
        mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f, mkl_A, descr, 
                        SPARSE_LAYOUT_ROW_MAJOR, B, k, k, 0.0f, C_mkl, k);
        end = get_time();
        printf("MKL CSC Time: %f s\n", end - start);
        mkl_sparse_destroy(mkl_A);
        // Custom CSC execution
        start = get_time();
        custom_csc_mm(&A, B, C_custom, k);
        end = get_time();
        printf("Custom CSC Time: %f s\n", end - start);
        free_csc(&A);
    }
    free(B);
    free(C_mkl);
    free(C_custom);
}
int main() {
    srand(time(NULL));
    for (int i = 0; i < 2; i++) {
        const int n_vals[] = {2048, 32768};
        const int n = n_vals[i];
        for (int j = 0; j < 2; j++) {
            const int k_vals[] = {64, 512};
            const int k = k_vals[j];
            // Test 3 shapes for A: (n,n), (4n,n), (n,4n)
            const int shapes[3][2] = {
                {n, n},
                {4*n, n},
                {n, 4*n}
            };
            for (int s = 0; s < 3; s++) {
                const int m = shapes[s][0];
                const int n_A = shapes[s][1];
                // CSR Format
                run_benchmark(m, n_A, k, 1);
                // CSC Format
                // run_benchmark(m, n_A, k, 0); // uncomment to test CSC
            }
        }
    }
    return 0;
}
