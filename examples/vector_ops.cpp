// vector_ops.cpp
// Demonstrates core Level-1 BLAS-style real-vector operations using theblas.
//
// Routines covered:
//   snrm2  - Euclidean norm
//   sscal  - in-place scalar multiply ("normalize")
//   saxpy  - y <- alpha*x + y
//   sdot   - dot product
//   isamax - index of element with largest absolute value

#include "theblas/theblas.h"

#include <cmath>
#include <cstdio>

int main() {
    // -----------------------------------------------------------------
    // 1. Normalize a float vector with snrm2 + sscal
    // -----------------------------------------------------------------
    float v[] = {3.0F, 4.0F, 0.0F}; // ||v|| = 5
    int n = 3;

    float norm = theblas::snrm2(n, v, 1);
    std::printf("Original vector:   [%.4f, %.4f, %.4f]\n", v[0], v[1], v[2]);
    std::printf("||v||              = %.4f\n", norm);

    theblas::sscal(n, 1.0F / norm, v, 1); // v <- v / ||v||
    std::printf("Normalized vector: [%.4f, %.4f, %.4f]\n", v[0], v[1], v[2]);
    std::printf("||v_norm||         = %.4f  (should be 1.0)\n\n",
                theblas::snrm2(n, v, 1));

    // -----------------------------------------------------------------
    // 2. Weighted combination with saxpy: result <- alpha*a + b
    // -----------------------------------------------------------------
    const float a[] = {1.0F, 2.0F, 3.0F};
    float b[] = {10.0F, 20.0F, 30.0F};
    float alpha = 0.5F;

    std::printf("a = [1, 2, 3],  b = [10, 20, 30],  alpha = %.1f\n", alpha);
    theblas::saxpy(n, alpha, a, 1, b, 1); // b <- 0.5*a + b
    std::printf("After saxpy (b <- alpha*a + b): [%.4f, %.4f, %.4f]\n\n",
                b[0], b[1], b[2]);

    // -----------------------------------------------------------------
    // 3. Dot product with sdot
    // -----------------------------------------------------------------
    const float x[] = {1.0F, 0.0F, -1.0F};
    const float y[] = {2.0F, 5.0F,  2.0F};

    float dot = theblas::sdot(n, x, 1, y, 1);
    std::printf("x = [1,  0, -1],  y = [2, 5,  2]\n");
    std::printf("x . y = %.4f  (should be 0.0 — orthogonal)\n\n", dot);

    // -----------------------------------------------------------------
    // 4. Find the index of the element with largest absolute value
    // -----------------------------------------------------------------
    const float w[] = {-1.0F, 4.0F, -7.0F, 2.0F}; // max |w[i]| is at index 3 (1-based)
    int imax = theblas::isamax(4, w, 1);      // returns 1-based index

    std::printf("w = [-1, 4, -7, 2]\n");
    std::printf("Index of max |w[i]| (1-based): %d  (should be 3)\n", imax);

    return 0;
}
