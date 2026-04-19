/**
 * @file matrix_vector_ops.cpp
 * @brief Demonstrates Level 2 BLAS routines: gemv, symv, trmv, trsv, ger.
 *
 * Build and run:
 *   cmake -S .. -B ../build && cmake --build ../build
 *   ../build/examples/example_matrix_vector_ops
 */

#include "theblas/theblas.h"

#include <array>
#include <cstdio>

int main() {
	// ---------------------------------------------------------------
	// sgemv — General matrix-vector multiply: y ← α·A·x + β·y
	// ---------------------------------------------------------------
	// A (3×2, column-major): [[1, 2],
	//                         [3, 4],
	//                         [5, 6]]
	{
		std::array<float, 6> A = {1, 3, 5, 2, 4, 6};
		std::array<float, 2> x = {1, 1};
		std::array<float, 3> y = {0, 0, 0};

		theblas::sgemv('N', 3, 2, 1.0F, A.data(), 3, x.data(), 1, 0.0F, y.data(), 1);

		std::printf("sgemv  y = [%.1f, %.1f, %.1f]\n", static_cast<double>(y[0]),
		            static_cast<double>(y[1]), static_cast<double>(y[2]));
		// Expected: [3.0, 7.0, 11.0]
	}

	// ---------------------------------------------------------------
	// dsymv — Symmetric matrix-vector multiply: y ← α·A·x + β·y
	// ---------------------------------------------------------------
	// A (2×2 symmetric, upper stored): [[2, 1],
	//                                   [1, 3]]
	{
		std::array<double, 4> A = {2, 0, 1, 3};
		std::array<double, 2> x = {1, 2};
		std::array<double, 2> y = {0, 0};

		theblas::dsymv('U', 2, 1.0, A.data(), 2, x.data(), 1, 0.0, y.data(), 1);

		std::printf("dsymv  y = [%.1f, %.1f]\n", y[0], y[1]);
		// Expected: [4.0, 7.0]
	}

	// ---------------------------------------------------------------
	// strmv — Triangular matrix-vector multiply: x ← A·x
	// ---------------------------------------------------------------
	// A (2×2 upper triangular): [[2, 3],
	//                            [0, 4]]
	{
		std::array<float, 4> A = {2, 0, 3, 4};
		std::array<float, 2> x = {1, 2};

		theblas::strmv('U', 'N', 'N', 2, A.data(), 2, x.data(), 1);

		std::printf("strmv  x = [%.1f, %.1f]\n", static_cast<double>(x[0]),
		            static_cast<double>(x[1]));
		// Expected: [8.0, 8.0]
	}

	// ---------------------------------------------------------------
	// strsv — Triangular solve: solve A·x = b
	// ---------------------------------------------------------------
	// Same A as above, b = [8, 8] → x = [1, 2]
	{
		std::array<float, 4> A = {2, 0, 3, 4};
		std::array<float, 2> x = {8, 8}; // right-hand side b

		theblas::strsv('U', 'N', 'N', 2, A.data(), 2, x.data(), 1);

		std::printf("strsv  x = [%.1f, %.1f]\n", static_cast<double>(x[0]),
		            static_cast<double>(x[1]));
		// Expected: [1.0, 2.0]
	}

	// ---------------------------------------------------------------
	// sger — Rank-1 update: A ← α·x·yᵀ + A
	// ---------------------------------------------------------------
	{
		std::array<float, 2> x = {1, 2};
		std::array<float, 2> y = {3, 4};
		std::array<float, 4> A = {0, 0, 0, 0};

		theblas::sger(2, 2, 1.0F, x.data(), 1, y.data(), 1, A.data(), 2);

		std::printf("sger   A = [[%.1f, %.1f], [%.1f, %.1f]]\n", static_cast<double>(A[0]),
		            static_cast<double>(A[2]), static_cast<double>(A[1]),
		            static_cast<double>(A[3]));
		// Expected: [[3.0, 4.0], [6.0, 8.0]]
	}

	return 0;
}
