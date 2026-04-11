#ifndef BLAS_TEST_UTILS_H
#define BLAS_TEST_UTILS_H
/**
 * Generate a row major SPD n*n matrix with packed form
 * @param A data desc
 * @param n matrix dim
 * @param k condition number, must greater than 1
 */
void d_gen_SPD(double *A, long long n, double k);

/**
 * print a packed form SPD matrix
 * @param A a packed form SPD matrix
 * @param n dim
 */
void d_print_SPD_matrix(double const *A, long long n);

void d_gen_vec(double *vec, long long n);

void d_print_vec(double const *vec, long long n);

/**
 * save linear system Ax = b into file
 * @param A a n*n SPD matrix with packed form
 * @param n dim
 * @param b nonhomogenous term for Ax = b
 * @param x sol. term for Ax = b
 * @param filename where to save
 */
void d_save_SPD_system(double const *A, long long n, double const *b, double const *x, char const *filename);

double timer(void);

#endif //BLAS_TEST_UTILS_H