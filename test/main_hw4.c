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
    MKL_INT nnz;
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
void custom_mm(const Matrix *A, const float *B, float *C, MKL_INT const k, int const use_csr) {
    // C = A * B
    memset(C, 0, A->m * k * sizeof(float));
    // B shape: A->n * k
    if (use_csr) {
        for (MKL_INT i = 0; i < A->m; i++) { // CSR 外層迴圈應該要是 A->m (列數)
            for (MKL_INT l = A->ptr[i]; l < A->ptr[i + 1]; l++) {
                MKL_INT const a_col = A->indices[l];
                float const a_val = A->values[l];
                for (MKL_INT j = 0; j < k; j++) {
                    C[i * k + j] += a_val * B[a_col * k + j];
                }
            }
        }
    }
    else {
        for (MKL_INT i = 0; i < A->n; i++) { // CSC 外層迴圈是 A->n (行數)
            for (MKL_INT l = A->ptr[i]; l < A->ptr[i + 1]; l++) {
                MKL_INT const a_row = A->indices[l];
                float const a_val = A->values[l];
                for (MKL_INT j = 0; j < k; j++) {
                    // C 的 row 是 a_row
                    // B 的 row 是 i (因為 A 的行要對到 B 的列)
                    C[a_row * k + j] += a_val * B[i * k + j];
                }
            }
        }
    }
}
void generate_random_matrix(Matrix *A, MKL_INT const row, MKL_INT const col, MKL_INT target_nnz, int const is_csr) {
    // ensure target_nnz is reasonable
    target_nnz = target_nnz > row * col ? row * col : target_nnz;
    A->m = row;
    A->n = col;
    A->nnz = target_nnz;
    A->values = (float *)malloc(target_nnz * sizeof(float));
    A->indices = (MKL_INT *)malloc(target_nnz * sizeof(MKL_INT));
    A->is_csr = is_csr;

    MKL_INT const compressed_dim = is_csr ? row : col, other_dim = is_csr ? col : row;
    A->ptr = (MKL_INT *)calloc(compressed_dim + 1, sizeof(MKL_INT));

    // Simple random distribution of nnz across rows or cols
    MKL_INT const nnz_per_cr = target_nnz / compressed_dim, remainder = target_nnz % compressed_dim;
    MKL_INT current_nnz = 0;
    MKL_INT *usages = malloc(other_dim * sizeof(MKL_INT));
    if (is_csr) {
        for (MKL_INT i = 0; i < row; i++) {
            A->ptr[i] = current_nnz;
            MKL_INT const row_nnz = nnz_per_cr + (i < remainder ? 1 : 0);
            memset(usages, 0, other_dim * sizeof(MKL_INT));
            for (MKL_INT j = 0; j < row_nnz; j++) {
                A->values[current_nnz] = (float)rand() / (float)RAND_MAX;
                MKL_INT index = rand() % other_dim;
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
        for (MKL_INT j = 0; j < col; j++) {
            A->ptr[j] = current_nnz;
            MKL_INT const col_nnz = nnz_per_cr + (j < remainder ? 1 : 0);
            memset(usages, 0, other_dim * sizeof(MKL_INT));
            for (MKL_INT i = 0; i < col_nnz; i++) {
                A->values[current_nnz] = (float)rand() / (float)RAND_MAX;
                MKL_INT index = rand() % other_dim;
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
void generate_dense(float *mat, MKL_INT const row, MKL_INT const col) {
    for (MKL_INT i = 0; i < row * col; i++) {
        mat[i] = (float)rand() / (float)RAND_MAX;
    }
}
void error_check(float const *A, float const *B, MKL_INT const row, MKL_INT const col, float const tol) {
    MKL_INT const N = row * col;
    float *diff = malloc(N * sizeof(float));
    cblas_scopy(N, A, 1, diff, 1);
    cblas_saxpy(N, -1.0f, B, 1, diff, 1);
    float const abs_err = cblas_snrm2(N, diff, 1);
    float const rel_err = abs_err / cblas_snrm2(N, A, 1);
    if (rel_err > tol) {
        printf("Error check FAILED: Rel err %e (Abs: %e) exceeds tol %e\n", rel_err, abs_err, tol);
    } else {
        printf("Error check PASSED: Rel err %e (Abs: %e) is within tol %e\n", rel_err, abs_err, tol);
    }
    free(diff);
}
void run_benchmark(MKL_INT const m, MKL_INT const n, MKL_INT const k, int const use_csr) {
    MKL_INT const target_nnz = 32 * m;
    printf("\nBenchmarking: A(%lld x %lld), B(%lld x %lld), target_nnz=%lld, format=%s\n",
           (long long)m, (long long)n, (long long)n, (long long)k, (long long)target_nnz, use_csr ? "CSR" : "CSC");
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
        mkl_sparse_set_mm_hint(mkl_A, SPARSE_OPERATION_NON_TRANSPOSE, descr,
            SPARSE_LAYOUT_ROW_MAJOR, k, 10000000);
        mkl_sparse_set_memory_hint(mkl_A, SPARSE_MEMORY_AGGRESSIVE);
        mkl_sparse_optimize(mkl_A);
        start = get_time();
        // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/mkl-sparse-mm.html
        mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f, mkl_A, descr, 
                        SPARSE_LAYOUT_ROW_MAJOR, B, k, k, 0.0f, C_mkl, k);
        end = get_time();
        printf("MKL CSR Time:\t\t%f s\n", end - start);
        mkl_sparse_destroy(mkl_A);
    } else {
        // MKL CSC execution
        mkl_sparse_s_create_csc(&mkl_A, SPARSE_INDEX_BASE_ZERO, A.m, A.n, 
                                A.ptr, A.ptr + 1, A.indices, A.values);
        mkl_sparse_set_mm_hint(mkl_A, SPARSE_OPERATION_NON_TRANSPOSE, descr,
            SPARSE_LAYOUT_ROW_MAJOR, k, 10000000);
        mkl_sparse_set_memory_hint(mkl_A, SPARSE_MEMORY_AGGRESSIVE);
        mkl_sparse_optimize(mkl_A);
        start = get_time();
        mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f, mkl_A, descr, 
                        SPARSE_LAYOUT_ROW_MAJOR, B, k, k, 0.0f, C_mkl, k);
        end = get_time();
        printf("MKL CSC Time:\t\t%f s\n", end - start);
        mkl_sparse_destroy(mkl_A);
    }
    // Custom CSR/CSC execution
    start = get_time();
    custom_mm(&A, B, C_custom, k, use_csr);
    end = get_time();
    printf("Custom %s Time:\t%f s\n", use_csr ? "CSR" : "CSC", end - start);
    // Check error against MKL result
    error_check(C_mkl, C_custom, m, k, 1e-7f);
    free_matrix(&A);
    free(B);
    free(C_mkl);
    free(C_custom);
}
int main() {
    srand(time(NULL));
    MKL_INT const n_vals[] = {2048, 32768}, k_vals[] = {64, 512};
    for (MKL_INT i = 0; i < 2; i++) {
        for (MKL_INT j = 0; j < 2; j++) {
            // Test 3 shapes for A: (n,n), (4n,n), (n,4n)
            MKL_INT const shapes[3][2] = {
                {n_vals[i], n_vals[i]},
                {4 * n_vals[i], n_vals[i]},
                {n_vals[i], 4 * n_vals[i]}
            };
            for (MKL_INT s = 0; s < 3; s++) {
                // CSR Format
                run_benchmark(shapes[s][0], shapes[s][1], k_vals[j], 1);
                // CSC Format
                run_benchmark(shapes[s][0], shapes[s][1], k_vals[j], 0);
            }
        }
    }
    return 0;
}
