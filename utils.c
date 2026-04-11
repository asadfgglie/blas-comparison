#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

#ifdef USE_NETLIB
#include <cblas.h>
#endif

double timer(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

void d_gen_SPD(double *A, long long const n, double k) {
    double *matrix = malloc(n * n * sizeof(double)), *result = malloc(n * n * sizeof(double));
    for (long long i = 0; i < n * n; i++) {
        matrix[i] = drand48() - 0.5;
    }
    memset(A, 0 , sizeof(double) * n * (n + 1) / 2);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, n, 1.0, matrix,
        n, matrix, n, 0.0, result, n);

    k = k > 1 ? k : 1 + 1e-10;
    double *vec = malloc(n * sizeof(double)), e = (double)n / (3 * (k - 1));
    for (long long i = 0; i < n; i++) {
        vec[i] = e;
    }

    cblas_daxpy(n, 1.0, vec, 1, result, n + 1);

    long long offset = 0;
    for (long long i = 0; i < n; i++) {
        cblas_dcopy(n - i, result + i * n + i, 1, A + offset, 1);
        offset += n - i;
    }

    free(vec);
    free(matrix);
    free(result);
}

void d_print_SPD_matrix(double const *A, long long const n) {
    for (long long i = 0; i < n; i++) {
        for (long long j = 0; j < n; j++) {
            const long long row = i <= j ? i : j;
            const long long col = i <= j ? j : i;
            const long long idx = row * n - row * (row - 1) / 2 + (col - row);
            printf("%.3f ", A[idx]);
        }
        printf("\n");
    }
}

void d_gen_vec(double *vec, long long const n) {
    for (long long i = 0; i < n; i++) {
        vec[i] = drand48();
    }
}

void d_print_vec(double const *vec, long long const n) {
    for (long long i = 0; i < n; i++) {
        printf("%.3f ", vec[i]);
    }
    printf("\n");
}

void d_save_SPD_system(double const *A, long long const n, double const *b, double const *x, char const *filename) {
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%lld\n", n);
    for (long long i = 0; i < n * (n + 1) / 2; i++) {
        fprintf(fp, "%.17f ", A[i]);
    }
    fprintf(fp, "\n");
    for (long long i = 0; i < n; i++) {
        fprintf(fp, "%.17f ", b[i]);
    }
    fprintf(fp, "\n");
    for (long long i = 0; i < n; i++) {
        fprintf(fp, "%.17f ", x[i]);
    }
}