#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "utils.h"

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

#ifdef USE_NETLIB
#include <cblas.h>
#endif

/**
 * use Steepest Descent to solve Ax = b
 * @param matrix n*n SPD matrix A with packed form
 * @param n dim
 * @param b nonhomogenous term for Ax = b
 * @param x sol.
 * @param tolerance tolerance range
 */
long long d_steepest_descent(double const *matrix, long long const n, double const *b, double *x, double const tolerance) {
    memset(x, 0, sizeof(double) * n);
    double e = INFINITY, *r = malloc(sizeof(double) * n);
    cblas_dcopy(n, b, 1, r, 1);
    double *y = malloc(n * sizeof(double));

    long long iter = 0;
    while (e > tolerance) {
        cblas_dspmv(CblasRowMajor, CblasUpper, n, 1.0, matrix, r, 1, 0.0, y, 1);
        const double alpha = cblas_ddot(n, r, 1, r, 1) / cblas_ddot(n, r, 1, y, 1);
        // alpha <- r^Tr/r^tAr

        cblas_daxpy(n, alpha, r, 1, x, 1);
        // x <- x + alpha * r

        cblas_dspmv(CblasRowMajor, CblasUpper, n, -1.0, matrix, x, 1, 0.0, y, 1);
        cblas_daxpy(n, 1.0, b, 1, y, 1);
        cblas_dcopy(n, y, 1, r, 1);
        // r <- -Ax + b

        e = cblas_dnrm2(n, r, 1);
        // e <- norm(r)

        iter++;
    }

    free(r);
    free(y);

    return iter;
}

int main(int const argc, char **argv) {
    srand48(time(NULL));

    char *filename = "./result.txt";
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
    double *A = malloc(sizeof(double) * n * (n + 1) / 2);
    d_gen_SPD(A, n, 10);
    double const end_gen_SPD = timer();
    if (verbose) {
        printf("SPD A matrix:\n");
        d_print_SPD_matrix(A, n);
    }
    printf("\ngen SPD timer: %.5fs\n", end_gen_SPD - start_gen_SPD);

    double const start_gen_vec = timer();
    double *b = malloc(sizeof(double) * n);
    d_gen_vec(b, n);
    double const end_gen_vec = timer();
    if (verbose) {
        printf("\nb vector:\n");
        d_print_vec(b, n);
    }
    printf("\ngen b vector timer: %.5fs\n", end_gen_vec - start_gen_vec);

    double *x = malloc(sizeof(double) * n);

    double const start = timer();
    long long const iter = d_steepest_descent(A, n, b, x, 1e-10);
    double const end = timer();
    if (verbose) {
        printf("\nsol. x vector:\n");
        d_print_vec(x, n);
    }
    printf("\nsteepest_descent timer: %.5fs, iter: %lld\n", end - start, iter);

    d_save_SPD_system(A, n, b, x, filename);
    printf("\nprecision linear system is save into `%s`.", filename);

    free(A);
    free(b);
    free(x);
    return 0;
}