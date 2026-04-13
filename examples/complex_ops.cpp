// complex_ops.cpp
// Demonstrates Level-1 BLAS-style complex-vector operations using theblas.
//
// Routines covered:
//   dznrm2  - Euclidean norm of complex-double vector
//   zdscal  - scale complex-double vector by a real scalar
//   zaxpy   - y <- alpha*x + y  (complex-double)
//   zdotc   - conjugated dot product (inner product) of complex-double vectors

#include "theblas/theblas.h"

#include <cmath>
#include <complex>
#include <cstdio>

int main() {
    using cd = std::complex<double>;

    // -----------------------------------------------------------------
    // 1. Normalize a complex-double vector with dznrm2 + zdscal
    // -----------------------------------------------------------------
    cd u[] = { {3.0, 0.0}, {0.0, 4.0} }; // ||u|| = 5
    int n = 2;

    double norm = theblas::dznrm2(n, u, 1);
    std::printf("Original: [(%.1f%+.1fi), (%.1f%+.1fi)]\n",
                u[0].real(), u[0].imag(), u[1].real(), u[1].imag());
    std::printf("||u|| = %.4f\n", norm);

    theblas::zdscal(n, 1.0 / norm, u, 1); // u <- u / ||u||
    std::printf("Normalized: [(%.4f%+.4fi), (%.4f%+.4fi)]\n",
                u[0].real(), u[0].imag(), u[1].real(), u[1].imag());
    std::printf("||u_norm|| = %.4f  (should be 1.0)\n\n",
                theblas::dznrm2(n, u, 1));

    // -----------------------------------------------------------------
    // 2. Superposition of two complex signals with zaxpy
    // -----------------------------------------------------------------
    cd signal_a[] = { {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0} };
    cd signal_b[] = { {0.0, 1.0}, {1.0, 0.0}, { 0.0, 1.0} };
    cd alpha = { 2.0, 0.0 }; // scale signal_a by 2 before adding
    int n3 = 3;

    std::printf("signal_a = [(1+0i), (0+1i), (-1+0i)]\n");
    std::printf("signal_b = [(0+1i), (1+0i),  (0+1i)]\n");
    std::printf("alpha = 2\n");

    theblas::zaxpy(n3, alpha, signal_a, 1, signal_b, 1); // signal_b <- 2*signal_a + signal_b
    std::printf("After zaxpy (signal_b <- alpha*signal_a + signal_b):\n");
    for (int i = 0; i < n3; ++i) {
        std::printf("  [%d] = (%.4f%+.4fi)\n", i, signal_b[i].real(), signal_b[i].imag());
    }
    std::puts("");

    // -----------------------------------------------------------------
    // 3. Complex inner product with zdotc (conjugates the first argument)
    //    <a, b> = sum_i conj(a[i]) * b[i] — returns real value for orthonormal bases
    // -----------------------------------------------------------------
    cd p[] = { {1.0, 0.0}, {0.0, 1.0} }; // unit vector along (1, i)
    cd q[] = { {0.0, 1.0}, {1.0, 0.0} }; // another unit vector

    cd inner = theblas::zdotc(n, p, 1, q, 1);
    std::printf("p = [(1+0i), (0+1i)]\n");
    std::printf("q = [(0+1i), (1+0i)]\n");
    std::printf("<p, q> = zdotc = (%.4f%+.4fi)\n", inner.real(), inner.imag());

    return 0;
}
