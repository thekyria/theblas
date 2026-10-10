/**
 * @file matrix_matrix_ops.cpp
 * @brief Demonstrates Level 3 matrix-matrix multiplication with sgemm.
 *
 * Build and run from the repository root:
 *   cmake -S . -B build
 *   cmake --build build
 *   build/examples/example_matrix_matrix_ops
 */

#include "theblas/theblas.h"

#include <cstdio>

int main() {
    // Column-major storage: element (i, j) is at i + j * ld.
    // A (2x3) = [[1, 2, 3], [4, 5, 6]]
    // B (3x2) = [[7, 8], [9, 10], [11, 12]]
    const float a[] = {1, 4, 2, 5, 3, 6};
    const float b[] = {7, 9, 11, 8, 10, 12};
    float c[4] = {};

    // C = alpha * A * B + beta * C; 'N' means no transpose.
    // m = 2, n = 2, k = 3; lda = 2, ldb = 3, ldc = 2 (stored row counts).
    theblas::sgemm('N', 'N', 2, 2, 3, 1.0F, a, 2, b, 3, 0.0F, c, 2);

    std::printf("sgemm  C = [[%.1f, %.1f], [%.1f, %.1f]]\n", static_cast<double>(c[0]),
                static_cast<double>(c[2]), static_cast<double>(c[1]), static_cast<double>(c[3]));
    // Expected: [[58.0, 64.0], [139.0, 154.0]].
    return (c[0] == 58.0F && c[1] == 139.0F && c[2] == 64.0F && c[3] == 154.0F) ? 0 : 1;
}
