import numpy as np
import argparse

import scipy

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="input file")

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.file, "r") as f:
        n = int(f.readline())
        A_packed = [] # packed SPD matrix with upper triangular part
        A_packed.extend(map(float, f.readline().split()))
        A = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i, n):
                A[i, j] = A_packed[idx]
                A[j, i] = A_packed[idx]
                idx += 1
        b = []
        b.extend(map(float, f.readline().split()))
        b = np.array(b, dtype=np.float64)
        computed_x = []
        computed_x.extend(map(float, f.readline().split()))
        computed_x = np.array(computed_x, dtype=np.float64)

    Ax = A @ computed_x
    residual = np.linalg.norm(Ax - b)
    is_ax_b_close = np.allclose(Ax, b, atol=1e-10)

    print(f"Residual ||A(computed_x) - b||: {residual:.5e}")
    print(f"if A(computed_x) are element-wise equal to scipy_x within a tolerance (tolerance < 1e-10): {is_ax_b_close}")

    scipy_x = scipy.linalg.solve(A, b, assume_a='pos')

    diff_x = np.linalg.norm(scipy_x - computed_x)
    is_x_close = np.allclose(scipy_x, computed_x, atol=1e-10)

    print(f"||scipy_x - computed_x||: {diff_x:.5e}")
    print(f"if computed_x are element-wise equal to scipy_x within a tolerance (tolerance < 1e-10): {is_x_close}")