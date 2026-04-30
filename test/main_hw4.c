#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl.h>
#include <string.h>
#include <sys/time.h>
// CSR/CSC format structure
typedef struct {
    MKL_INT m; // row
    MKL_INT n; // col
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
void generate_random_matrix(Matrix *A, int const row, int const col, int target_nnz, int const is_csr) {
    // ensure target_nnz is reasonable
    target_nnz = target_nnz > row * col ? row * col : target_nnz;
    A->m = row;
    A->n = col;
    A->nnz = target_nnz;
    A->values = (float *)malloc(target_nnz * sizeof(float));
    A->indices = (MKL_INT *)malloc(target_nnz * sizeof(MKL_INT));
    A->is_csr = is_csr;

    int const compressed_dim = is_csr ? row : col, other_dim = is_csr ? col : row;
    A->ptr = (MKL_INT *)calloc(compressed_dim + 1, sizeof(MKL_INT));

    // Simple random distribution of nnz across rows or cols
    int const nnz_per_cr = target_nnz / compressed_dim, remainder = target_nnz % compressed_dim;
    int current_nnz = 0;
    int *usages = malloc(other_dim * sizeof(int));
    if (is_csr) {
        for (int i = 0; i < row; i++) {
            A->ptr[i] = current_nnz;
            int const row_nnz = nnz_per_cr + (i < remainder ? 1 : 0);
            memset(usages, 0, other_dim * sizeof(int));
            for (int j = 0; j < row_nnz; j++) {
                A->values[current_nnz] = (float)rand() / (float)RAND_MAX;
                int index = rand() % other_dim;
                while (usages[index] > 0) {
                    index = rand() % other_dim;
                }
                usages[index]++;
                A->indices[current_nnz] = index;
                current_nnz++;
            }
        }
    }
    else {
        for (int j = 0; j < col; j++) {
            A->ptr[j] = current_nnz;
            int const col_nnz = nnz_per_cr + (j < remainder ? 1 : 0);
            memset(usages, 0, other_dim * sizeof(int));
            for (int i = 0; i < col_nnz; i++) {
                A->values[current_nnz] = (float)rand() / (float)RAND_MAX;
                int index = rand() % other_dim;
                while (usages[index] > 0) {
                    index = rand() % other_dim;
                }
                usages[index]++;
                A->indices[current_nnz] = index;
                current_nnz++;
            }
        }
    }
    A->ptr[compressed_dim] = current_nnz;

    free(usages);
}
void free_matrix(Matrix const *A) {
    free(A->values);
    free(A->indices);
    free(A->ptr);
}
void generate_dense(float *mat, int const row, int const col) {
    if (mat == NULL) {
        return;
    }
    for (int i = 0; i < row * col; i++) {
        mat[i] = (float)rand() / (float)RAND_MAX;
    }
}
void run_benchmark(int const m, int const n, int const k, int const use_csr) {
    int const target_nnz = 32 * m;
    printf("\nBenchmarking: A(%d x %d), B(%d x %d), target_nnz=%d, format=%s\n", 
           m, n, n, k, target_nnz, use_csr ? "CSR" : "CSC");
    float *B = malloc(n * k * sizeof(float));
    float *C_mkl = malloc(m * k * sizeof(float));
    float *C_custom = malloc(m * k * sizeof(float));
    generate_dense(B, n, k);
    double start, end;
    Matrix A;
    generate_random_matrix(&A, m, n, target_nnz, use_csr);
    sparse_matrix_t mkl_A;
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    if (use_csr) {
        // MKL CSR execution
        mkl_sparse_s_create_csr(&mkl_A, SPARSE_INDEX_BASE_ZERO, A.m, A.n, 
                                A.ptr, A.ptr + 1, A.indices, A.values);
        mkl_sparse_optimize(mkl_A);
        start = get_time();
        // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/mkl-sparse-mm.html
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
        // MKL CSC execution
        mkl_sparse_s_create_csc(&mkl_A, SPARSE_INDEX_BASE_ZERO, A.m, A.n, 
                                A.ptr, A.ptr + 1, A.indices, A.values);
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
    int const n_vals[] = {2048, 32768}, k_vals[] = {64, 512};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            // Test 3 shapes for A: (n,n), (4n,n), (n,4n)
            int const shapes[3][2] = {
                {n_vals[i], n_vals[i]},
                {4 * n_vals[i], n_vals[i]},
                {n_vals[i], 4 * n_vals[i]}
            };
            for (int s = 0; s < 3; s++) {
                int const m = shapes[s][0];
                int const n = shapes[s][1];
                // CSR Format
                run_benchmark(m, n, k_vals[j], 1);
                // CSC Format
                run_benchmark(m, n, k_vals[j], 0);
            }
        }
    }
    return 0;
}
