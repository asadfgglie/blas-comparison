#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl.h>
#include <sys/time.h>
// CSR format structure
typedef struct {
    MKL_INT m;
    MKL_INT n;
    int nnz;
    float *values;
    MKL_INT *indices;
    MKL_INT *ptr;
    int is_csr; // 1 if is csr, 0 if is csc
} Matrix;
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}
// TODO: Implement your custom CSR matrix multiplication here
void custom_csr_mm(const Matrix *A, const float *B, float *C, int k) {
    // Student to implement
    // C = A * B
}
// TODO: Implement your custom CSC matrix multiplication here
void custom_csc_mm(const Matrix *A, const float *B, float *C, int k) {
    // Student to implement
    // C = A * B
}
void generate_random_matrix(Matrix *A, int m, int n, int target_nnz, int is_csr) {
    A->m = m;
    A->n = n;
    A->nnz = target_nnz;
    A->values = (float *)malloc(target_nnz * sizeof(float));
    A->indices = (MKL_INT *)malloc(target_nnz * sizeof(MKL_INT));

    int dim = is_csr ? m : n;
    A->ptr = (MKL_INT *)calloc(dim + 1, sizeof(MKL_INT));

    // Simple random distribution of nnz across rows or cols
    int const nnz_per_cr = target_nnz / dim;
    int const remainder = target_nnz % dim;
    int current_nnz = 0;
    if (is_csr) {
        for (int i = 0; i < m; i++) {
            A->ptr[i] = current_nnz;
            int row_nnz = nnz_per_cr + (i < remainder ? 1 : 0);
            // To ensure we don't exceed columns, cap row_nnz to n
            if (row_nnz > n) row_nnz = n;
            for (int j = 0; j < row_nnz; j++) {
                A->values[current_nnz] = rand() / (float)RAND_MAX;
                A->indices[current_nnz] = (j * (n / row_nnz)) % n; // simplistic distinct columns
                current_nnz++;
            }
        }
    }
    else {
        for (int j = 0; j < n; j++) {
            A->ptr[j] = current_nnz;
            int col_nnz = nnz_per_cr + (j < remainder ? 1 : 0);
            if (col_nnz > m) col_nnz = m;
            for (int i = 0; i < col_nnz; i++) {
                A->values[current_nnz] = rand() / (float)RAND_MAX;
                A->indices[current_nnz] = (i * (m / col_nnz)) % m;
                current_nnz++;
            }
        }
    }
    A->ptr[dim] = current_nnz;

    // Fix actual nnz if capped
    A->nnz = current_nnz;
    A->is_csr = is_csr;
}
void free_matrix(Matrix *A) {
    free(A->values);
    free(A->indices);
    free(A->ptr);
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
    Matrix A;
    if (use_csr) {
        generate_random_matrix(&A, m, n_A, target_nnz, 1);
        // MKL CSR execution
        sparse_matrix_t mkl_A;
        mkl_sparse_s_create_csr(&mkl_A, SPARSE_INDEX_BASE_ZERO, A.m, A.n, 
                                A.ptr, A.ptr + 1, A.indices, A.values);
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
        free_matrix(&A);
    } else {
        generate_random_matrix(&A, m, n_A, target_nnz, 0);
        // MKL CSC execution
        sparse_matrix_t mkl_A;
        mkl_sparse_s_create_csc(&mkl_A, SPARSE_INDEX_BASE_ZERO, A.m, A.n, 
                                A.ptr, A.ptr + 1, A.indices, A.values);
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
        free_matrix(&A);
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
