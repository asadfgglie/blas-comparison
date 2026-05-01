#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl.h>
#include <string.h>
#include <sys/time.h>
#ifdef USE_OMP
#include <omp.h>
#endif
typedef struct { // CSR/CSC format structure
    MKL_INT m, n, nnz; /* row, col, nnz */ float *values;
    MKL_INT *indices, *ptr; int is_csr; // 1 if is csr, 0 if is csc
} Matrix;
double get_time() {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}
void custom_mm(const Matrix *A, const float *B, float *C, MKL_INT const k, int const use_csr) {
    memset(C, 0, A->m * k * sizeof(float));
    if (use_csr) {
#ifdef USE_OMP
        #pragma omp parallel for default(none) shared(A, B, C, k) schedule(static)
#endif
        for (MKL_INT i = 0; i < A->m; i++) {
            for (MKL_INT l = A->ptr[i]; l < A->ptr[i + 1]; l++) {
                MKL_INT const a_col = A->indices[l]; float const a_val = A->values[l];
#ifdef USE_OMP
                #pragma omp simd
#endif
                for (MKL_INT j = 0; j < k; j++) { C[i * k + j] += a_val * B[a_col * k + j]; }
            }
        }
    }
    else {
#ifdef USE_OMP
        #pragma omp parallel for default(none) shared(A, B, C, k) schedule(static)
#endif
        for (MKL_INT j = 0; j < k; j++) {
            for (MKL_INT i = 0; i < A->n; i++) {
                for (MKL_INT l = A->ptr[i]; l < A->ptr[i + 1]; l++) {
                    MKL_INT const a_row = A->indices[l]; float const a_val = A->values[l];
                    C[a_row * k + j] += a_val * B[i * k + j];
                }
            }
        }
    }
}
int cmp (const void * a, const void * b) { return *(MKL_INT*)a > *(MKL_INT*)b; }
void generate_random_matrix(Matrix *A, MKL_INT const row, MKL_INT const col, MKL_INT target_nnz, int const is_csr) {
    // ensure target_nnz is reasonable
    target_nnz = target_nnz > row * col ? row * col : target_nnz;
    A->m = row; A->n = col; A->nnz = target_nnz; A->is_csr = is_csr;
    A->values = (float *)mkl_malloc(target_nnz * sizeof(float), 64);
    A->indices = (MKL_INT *)mkl_malloc(target_nnz * sizeof(MKL_INT), 64);
    MKL_INT const compressed_dim = is_csr ? row : col, other_dim = is_csr ? col : row;
    A->ptr = (MKL_INT *)mkl_malloc((compressed_dim + 1) * sizeof(MKL_INT), 64);
    MKL_INT const nnz_per_cr = target_nnz / compressed_dim, remainder = target_nnz % compressed_dim;
    MKL_INT current_nnz = 0;
    MKL_INT *usages = malloc(other_dim * sizeof(MKL_INT));
    if (is_csr) {
        for (MKL_INT i = 0; i < row; i++) {
            A->ptr[i] = current_nnz;
            MKL_INT const row_nnz = nnz_per_cr + (i < remainder ? 1 : 0);
            memset(usages, 0, other_dim * sizeof(MKL_INT));
            MKL_INT const current_start = current_nnz;
            for (MKL_INT j = 0; j < row_nnz; j++) {
                A->values[current_nnz] = (float)rand() / (float)RAND_MAX;
                MKL_INT index = rand() % other_dim;
                while (usages[index] > 0) { index = rand() % other_dim; }
                usages[index]++; A->indices[current_nnz] = index; current_nnz++;
            }
            qsort(A->indices + current_start, row_nnz, sizeof(MKL_INT), cmp);
        }
    }
    else {
        for (MKL_INT j = 0; j < col; j++) {
            A->ptr[j] = current_nnz;
            MKL_INT const col_nnz = nnz_per_cr + (j < remainder ? 1 : 0);
            memset(usages, 0, other_dim * sizeof(MKL_INT));
            MKL_INT const current_start = current_nnz;
            for (MKL_INT i = 0; i < col_nnz; i++) {
                A->values[current_nnz] = (float)rand() / (float)RAND_MAX;
                MKL_INT index = rand() % other_dim;
                while (usages[index] > 0) { index = rand() % other_dim; }
                usages[index]++; A->indices[current_nnz] = index; current_nnz++;
            }
            qsort(A->indices + current_start, col_nnz, sizeof(MKL_INT), cmp);
        }
    }
    A->ptr[compressed_dim] = current_nnz;
    free(usages);
}
void free_matrix(Matrix const *A) { mkl_free(A->values); mkl_free(A->indices); mkl_free(A->ptr); }
void generate_dense(float *mat, MKL_INT const row, MKL_INT const col) {
    for (MKL_INT i = 0; i < row * col; i++) { mat[i] = (float)rand() / (float)RAND_MAX; }
}
void error_check(float const *A, float const *B, MKL_INT const row, MKL_INT const col, float const tol) {
    MKL_INT const N = row * col; float *diff = malloc(N * sizeof(float));
    cblas_scopy(N, A, 1, diff, 1); cblas_saxpy(N, -1.0f, B, 1, diff, 1);
    float const abs_err = cblas_snrm2(N, diff, 1), rel_err = abs_err / cblas_snrm2(N, A, 1);
    if (rel_err > tol) {
        printf("Error check FAILED: Rel err %e (Abs: %e) exceeds tol %e\n", rel_err, abs_err, tol);
    } else {
        printf("Error check PASSED: Rel err %e (Abs: %e) is within tol %e\n", rel_err, abs_err, tol);
    }
    free(diff);
}
void save_result(Matrix const A, float const *B, float const *C_custom, float const *C_mkl, MKL_INT const k, char const *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return;
    }
    fprintf(fp, "=================================================\n");
    fprintf(fp, "A: m=%lld, n=%lld, nnz=%lld, is_csr=%d, k=%lld\n", (long long)A.m, (long long)A.n, (long long)A.nnz, A.is_csr, (long long)k);

    // 只印出左上角 10x10 的子矩陣，以避免檔案過大 (如果原始矩陣太大)
    MKL_INT print_m = A.m > 10 ? 10 : A.m;
    MKL_INT print_n = A.n > 10 ? 10 : A.n;
    MKL_INT print_k = k > 10 ? 10 : k;

    // 將 A 轉換成 Dense 的 10x10 方便觀察
    fprintf(fp, "\n--- Top-Left %lldx%lld of Dense A ---\n", (long long)print_m, (long long)print_n);
    float *dense_A = calloc(print_m * print_n, sizeof(float));
    if (A.is_csr) {
        for (MKL_INT i = 0; i < print_m; i++) {
            for (MKL_INT l = A.ptr[i]; l < A.ptr[i + 1]; l++) {
                MKL_INT col = A.indices[l];
                if (col < print_n) dense_A[i * print_n + col] = A.values[l];
            }
        }
    } else {
        for (MKL_INT j = 0; j < print_n; j++) {
            for (MKL_INT l = A.ptr[j]; l < A.ptr[j + 1]; l++) {
                MKL_INT row = A.indices[l];
                if (row < print_m) dense_A[row * print_n + j] = A.values[l];
            }
        }
    }
    for (MKL_INT i = 0; i < print_m; i++) {
        for (MKL_INT j = 0; j < print_n; j++) {
            fprintf(fp, "%8.4f ", dense_A[i * print_n + j]);
        }
        fprintf(fp, "\n");
    }
    free(dense_A);

    fprintf(fp, "\n--- Top-Left %lldx%lld of B ---\n", (long long)print_n, (long long)print_k);
    for (MKL_INT i = 0; i < print_n; i++) {
        for (MKL_INT j = 0; j < print_k; j++) {
            fprintf(fp, "%8.4f ", B[i * k + j]);
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n--- Top-Left %lldx%lld of C_custom ---\n", (long long)print_m, (long long)print_k);
    for (MKL_INT i = 0; i < print_m; i++) {
        for (MKL_INT j = 0; j < print_k; j++) {
            fprintf(fp, "%8.4f ", C_custom[i * k + j]);
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n--- Top-Left %lldx%lld of C_mkl ---\n", (long long)print_m, (long long)print_k);
    for (MKL_INT i = 0; i < print_m; i++) {
        for (MKL_INT j = 0; j < print_k; j++) {
            fprintf(fp, "%8.4f ", C_mkl[i * k + j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
    fclose(fp);
}
void run_benchmark(MKL_INT const m, MKL_INT const n, MKL_INT const k, int const use_csr) {
    MKL_INT const target_nnz = 32 * m;
    printf("\nBenchmarking: A(%lld x %lld), B(%lld x %lld), target_nnz=%lld, format=%s\n",
           (long long)m, (long long)n, (long long)n, (long long)k, (long long)target_nnz, use_csr ? "CSR" : "CSC");
    float *B = mkl_malloc(n * k * sizeof(float), 64), *C_mkl = mkl_malloc(m * k * sizeof(float), 64);
    float *C_custom = mkl_malloc(m * k * sizeof(float), 64);
    generate_dense(B, n, k);
    double start, end;
    Matrix A;
    generate_random_matrix(&A, m, n, target_nnz, use_csr);
    sparse_matrix_t mkl_A;
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    if (use_csr) {
        mkl_sparse_s_create_csr(&mkl_A, SPARSE_INDEX_BASE_ZERO, A.m, A.n, 
                                A.ptr, A.ptr + 1, A.indices, A.values);
        mkl_sparse_set_mm_hint(mkl_A, SPARSE_OPERATION_NON_TRANSPOSE, descr,
            SPARSE_LAYOUT_ROW_MAJOR, k, 10000000);
        mkl_sparse_set_memory_hint(mkl_A, SPARSE_MEMORY_AGGRESSIVE);
        mkl_sparse_optimize(mkl_A);
        start = get_time();
        mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f, mkl_A, descr, 
                        SPARSE_LAYOUT_ROW_MAJOR, B, k, k, 0.0f, C_mkl, k);
        end = get_time();
        printf("MKL CSR Time:\t\t%f s\n", end - start);
        mkl_sparse_destroy(mkl_A);
    }
    else {
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
    start = get_time();
    custom_mm(&A, B, C_custom, k, use_csr);
    end = get_time();
    printf("Custom %s Time:\t%f s\n", use_csr ? "CSR" : "CSC", end - start);
    error_check(C_mkl, C_custom, m, k, 1e-7f);
    // save_result(A, B, C_custom, C_mkl, k, "/home/asadfgglie/blas-test/test/result.txt");
    free_matrix(&A); mkl_free(B); mkl_free(C_mkl); mkl_free(C_custom);
}
int main() {
    srand(time(NULL));
    MKL_INT const n_vals[] = {2048, 32768}, k_vals[] = {64, 512};
    for (MKL_INT i = 0; i < 2; i++) {
        MKL_INT const shapes[3][2] = {
            {n_vals[i], n_vals[i]}, {4 * n_vals[i], n_vals[i]},
            {n_vals[i], 4 * n_vals[i]}
        };
        for (MKL_INT j = 0; j < 2; j++) {
            for (MKL_INT s = 0; s < 3; s++) {
                run_benchmark(shapes[s][0], shapes[s][1], k_vals[j], 1);
                run_benchmark(shapes[s][0], shapes[s][1], k_vals[j], 0);
            }
        }
    }
    return 0;
}
