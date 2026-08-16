#include "theblas/theblas.h"

#include "theblas_detail.hpp"

#include <algorithm>
#include <complex>

namespace theblas {

using detail::conj_val;
using detail::start_index;
using detail::to_upper;
using detail::valid_diag;
using detail::valid_trans;
using detail::valid_uplo;

/* ====================================================================
 * Level 2 BLAS — Matrix-Vector Operations
 * ==================================================================== */

namespace {

/* ------------------------------------------------------------------ */
/* gemv_impl — y ← α·op(a)·x + β·y                                  */
/* ------------------------------------------------------------------ */
template <typename T>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void gemv_impl(char trans, int m, int n, T alpha, const T* a, int lda, const T* x, int incx,
               T beta, T* y, int incy) {
	char tr = to_upper(trans);
	if (!valid_trans(trans) || m <= 0 || n <= 0 || incx == 0 || incy == 0) {
		return;
	}
	int leny = (tr == 'N') ? m : n;
	int lenx = (tr == 'N') ? n : m;

	// Scale y by beta
	{
		int iy = start_index(leny, incy);
		if (beta == T(0)) {
			for (int i = 0; i < leny; ++i) {
				y[iy] = T(0);
				iy += incy;
			}
		} else if (beta != T(1)) {
			for (int i = 0; i < leny; ++i) {
				y[iy] *= beta;
				iy += incy;
			}
		}
	}

	if (alpha == T(0)) {
		return;
	}

	if (tr == 'N') {
		// y ← α·a·x + y
		int jx = start_index(lenx, incx);
		for (int j = 0; j < n; ++j) {
			T temp = alpha * x[jx];
			int iy = start_index(leny, incy);
			for (int i = 0; i < m; ++i) {
                            y[iy] += temp * a[i + (j * lda)];
                            iy += incy;
			}
			jx += incx;
		}
	} else if (tr == 'T') {
		// y ← α·Aᵀ·x + y
		int jy = start_index(leny, incy);
		for (int j = 0; j < n; ++j) {
			T temp = T(0);
			int ix = start_index(lenx, incx);
			for (int i = 0; i < m; ++i) {
                            temp += a[i + (j * lda)] * x[ix];
                            ix += incx;
			}
			y[jy] += alpha * temp;
			jy += incy;
		}
	} else {
		// tr == 'C': y ← α·Aᴴ·x + y (conjugate-transpose)
		int jy = start_index(leny, incy);
		for (int j = 0; j < n; ++j) {
			T temp = T(0);
			int ix = start_index(lenx, incx);
			for (int i = 0; i < m; ++i) {
                            temp += conj_val(a[i + (j * lda)]) * x[ix];
                            ix += incx;
			}
			y[jy] += alpha * temp;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* symv_impl — y ← α·a·x + β·y   (a symmetric)                     */
/* ------------------------------------------------------------------ */
template <typename T>
void symv_impl(char uplo, int n, T alpha, const T* a, int lda, const T* x, int incx, T beta,
               T* y, int incy) {
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0) {
		return;
	}

	// Scale y by beta
	{
		int iy = start_index(n, incy);
		if (beta == T(0)) {
			for (int i = 0; i < n; ++i) {
				y[iy] = T(0);
				iy += incy;
			}
		} else if (beta != T(1)) {
			for (int i = 0; i < n; ++i) {
				y[iy] *= beta;
				iy += incy;
			}
		}
	}

	if (alpha == T(0)) {
		return;
	}

	if (ul == 'U') {
		int jx = start_index(n, incx);
		int jy = start_index(n, incy);
		for (int j = 0; j < n; ++j) {
			T temp1 = alpha * x[jx];
			T temp2 = T(0);
			int ix = start_index(n, incx);
			int iy = start_index(n, incy);
			for (int i = 0; i < j; ++i) {
                            y[iy] += temp1 * a[i + (j * lda)];
                            temp2 += a[i + (j * lda)] * x[ix];
                            ix += incx;
                            iy += incy;
			}
                        y[jy] += (temp1 * a[j + (j * lda)]) + (alpha * temp2);
                        jx += incx;
			jy += incy;
		}
	} else {
		int jx = start_index(n, incx);
		int jy = start_index(n, incy);
		for (int j = 0; j < n; ++j) {
			T temp1 = alpha * x[jx];
			T temp2 = T(0);
                        y[jy] += temp1 * a[j + (j * lda)];
                        int ix = jx;
			int iy = jy;
			for (int i = j + 1; i < n; ++i) {
				ix += incx;
				iy += incy;
                                y[iy] += temp1 * a[i + (j * lda)];
                                temp2 += a[i + (j * lda)] * x[ix];
                        }
			y[jy] += alpha * temp2;
			jx += incx;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* hemv_impl — y ← α·a·x + β·y   (a Hermitian)                     */
/* ------------------------------------------------------------------ */
template <typename T>
void hemv_impl(char uplo, int n, std::complex<T> alpha, const std::complex<T>* a, int lda,
               const std::complex<T>* x, int incx, std::complex<T> beta, std::complex<T>* y,
               int incy) {
	using C = std::complex<T>;
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0) {
		return;
	}

	// Scale y by beta
	{
		int iy = start_index(n, incy);
		if (beta == C(0)) {
			for (int i = 0; i < n; ++i) {
				y[iy] = C(0);
				iy += incy;
			}
		} else if (beta != C(1)) {
			for (int i = 0; i < n; ++i) {
				y[iy] *= beta;
				iy += incy;
			}
		}
	}

	if (alpha == C(0)) {
		return;
	}

	if (ul == 'U') {
		int jx = start_index(n, incx);
		int jy = start_index(n, incy);
		for (int j = 0; j < n; ++j) {
			C temp1 = alpha * x[jx];
			C temp2 = C(0);
			int ix = start_index(n, incx);
			int iy = start_index(n, incy);
			for (int i = 0; i < j; ++i) {
                            y[iy] += temp1 * a[i + (j * lda)];
                            temp2 += std::conj(a[i + (j * lda)]) * x[ix];
                            ix += incx;
                            iy += incy;
			}
			// Diagonal of a Hermitian matrix is real
                        y[jy] += (temp1 * T(a[j + (j * lda)].real())) + (alpha * temp2);
                        jx += incx;
			jy += incy;
		}
	} else {
		int jx = start_index(n, incx);
		int jy = start_index(n, incy);
		for (int j = 0; j < n; ++j) {
			C temp1 = alpha * x[jx];
			C temp2 = C(0);
                        y[jy] += temp1 * T(a[j + (j * lda)].real());
                        int ix = jx;
			int iy = jy;
			for (int i = j + 1; i < n; ++i) {
				ix += incx;
				iy += incy;
                                y[iy] += temp1 * a[i + (j * lda)];
                                temp2 += std::conj(a[i + (j * lda)]) * x[ix];
                        }
			y[jy] += alpha * temp2;
			jx += incx;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* trmv_impl — x ← op(a)·x   (a triangular)                        */
/* ------------------------------------------------------------------ */
template <typename T>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void trmv_impl(char uplo, char trans, char diag, int n, const T* a, int lda, T* x, int incx) {
	char ul = to_upper(uplo);
	char tr = to_upper(trans);
	char dg = to_upper(diag);
	if (!valid_uplo(uplo) || !valid_trans(trans) || !valid_diag(diag) || n <= 0 || incx == 0) {
		return;
	}
	bool unit = (dg == 'U');

	if (tr == 'N') {
		if (ul == 'U') {
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				if (x[jx] != T(0)) {
					T temp = x[jx];
					int ix = start_index(n, incx);
					for (int i = 0; i < j; ++i) {
                                            x[ix] += temp * a[i + (j * lda)];
                                            ix += incx;
					}
					if (!unit) {
                                            x[jx] *= a[j + (j * lda)];
                                        }
				}
				jx += incx;
			}
		} else {
                    int jx = start_index(n, incx) + ((n - 1) * incx);
                    for (int j = n - 1; j >= 0; --j) {
                        if (x[jx] != T(0)) {
                            T temp = x[jx];
                            int ix = start_index(n, incx) + ((n - 1) * incx);
                            for (int i = n - 1; i > j; --i) {
                                x[ix] += temp * a[i + (j * lda)];
                                ix -= incx;
                            }
                            if (!unit) {
                                x[jx] *= a[j + (j * lda)];
                            }
                        }
                        jx -= incx;
                    }
		}
	} else {
		// Transpose or conjugate-transpose
		bool conj = (tr == 'C');
		if (ul == 'U') {
                    int jx = start_index(n, incx) + ((n - 1) * incx);
                    for (int j = n - 1; j >= 0; --j) {
                        T temp = x[jx];
                        if (!unit) {
                            temp *= conj ? conj_val(a[j + (j * lda)]) : a[j + (j * lda)];
                        }
                        int ix = jx;
                        for (int i = j - 1; i >= 0; --i) {
                            ix -= incx;
                            temp += (conj ? conj_val(a[i + (j * lda)]) : a[i + (j * lda)]) * x[ix];
                        }
                        x[jx] = temp;
                        jx -= incx;
                    }
		} else {
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				T temp = x[jx];
				if (!unit) {
                                    temp *= conj ? conj_val(a[j + (j * lda)]) : a[j + (j * lda)];
                                }
				int ix = jx;
				for (int i = j + 1; i < n; ++i) {
					ix += incx;
                                        temp +=
                                            (conj ? conj_val(a[i + (j * lda)]) : a[i + (j * lda)]) *
                                            x[ix];
                                }
				x[jx] = temp;
				jx += incx;
			}
		}
	}
}

/* ------------------------------------------------------------------ */
/* trsv_impl — solve op(a)·x = b   (a triangular)                   */
/* ------------------------------------------------------------------ */
template <typename T>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void trsv_impl(char uplo, char trans, char diag, int n, const T* a, int lda, T* x, int incx) {
	char ul = to_upper(uplo);
	char tr = to_upper(trans);
	char dg = to_upper(diag);
	if (!valid_uplo(uplo) || !valid_trans(trans) || !valid_diag(diag) || n <= 0 || incx == 0) {
		return;
	}
	bool unit = (dg == 'U');

	if (tr == 'N') {
		if (ul == 'U') {
			// Back-substitution
                        int jx = start_index(n, incx) + ((n - 1) * incx);
                        for (int j = n - 1; j >= 0; --j) {
				if (!unit) {
                                    x[jx] /= a[j + (j * lda)];
                                }
				T temp = x[jx];
				int ix = jx;
				for (int i = j - 1; i >= 0; --i) {
					ix -= incx;
                                        x[ix] -= temp * a[i + (j * lda)];
                                }
				jx -= incx;
			}
		} else {
			// Forward-substitution
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				if (!unit) {
                                    x[jx] /= a[j + (j * lda)];
                                }
				T temp = x[jx];
				int ix = jx;
				for (int i = j + 1; i < n; ++i) {
					ix += incx;
                                        x[ix] -= temp * a[i + (j * lda)];
                                }
				jx += incx;
			}
		}
	} else {
		bool conj = (tr == 'C');
		if (ul == 'U') {
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				T temp = x[jx];
				int ix = start_index(n, incx);
				for (int i = 0; i < j; ++i) {
                                    temp -= (conj ? conj_val(a[i + (j * lda)]) : a[i + (j * lda)]) *
                                            x[ix];
                                    ix += incx;
				}
				if (!unit) {
                                    temp /= conj ? conj_val(a[j + (j * lda)]) : a[j + (j * lda)];
                                }
				x[jx] = temp;
				jx += incx;
			}
		} else {
                    int jx = start_index(n, incx) + ((n - 1) * incx);
                    for (int j = n - 1; j >= 0; --j) {
                        T temp = x[jx];
                        int ix = start_index(n, incx) + ((n - 1) * incx);
                        for (int i = n - 1; i > j; --i) {
                            temp -= (conj ? conj_val(a[i + (j * lda)]) : a[i + (j * lda)]) * x[ix];
                            ix -= incx;
                        }
                        if (!unit) {
                            temp /= conj ? conj_val(a[j + (j * lda)]) : a[j + (j * lda)];
                        }
                        x[jx] = temp;
                        jx -= incx;
                    }
		}
	}
}

/* ------------------------------------------------------------------ */
/* ger_impl — a ← α·x·yᵀ + a   (real rank-1 update)                */
/* ------------------------------------------------------------------ */
template <typename T>
void ger_impl(int m, int n, T alpha, const T* x, int incx, const T* y, int incy, T* a, int lda) {
	if (m <= 0 || n <= 0 || incx == 0 || incy == 0) {
		return;
	}
	if (alpha == T(0)) {
		return;
	}

	int jy = start_index(n, incy);
	for (int j = 0; j < n; ++j) {
		T temp = alpha * y[jy];
		int ix = start_index(m, incx);
		for (int i = 0; i < m; ++i) {
                    a[i + (j * lda)] += x[ix] * temp;
                    ix += incx;
		}
		jy += incy;
	}
}

/* ------------------------------------------------------------------ */
/* geru_impl — a ← α·x·yᵀ + a   (complex unconjugated rank-1)      */
/* ------------------------------------------------------------------ */
template <typename T>
void geru_impl(int m, int n, std::complex<T> alpha, const std::complex<T>* x, int incx,
               const std::complex<T>* y, int incy, std::complex<T>* a, int lda) {
	using C = std::complex<T>;
	if (m <= 0 || n <= 0 || incx == 0 || incy == 0) {
		return;
	}
	if (alpha == C(0)) {
		return;
	}

	int jy = start_index(n, incy);
	for (int j = 0; j < n; ++j) {
		C temp = alpha * y[jy];
		int ix = start_index(m, incx);
		for (int i = 0; i < m; ++i) {
                    a[i + (j * lda)] += x[ix] * temp;
                    ix += incx;
		}
		jy += incy;
	}
}

/* ------------------------------------------------------------------ */
/* gerc_impl — a ← α·x·conj(y)ᵀ + a   (complex conjugated rank-1)  */
/* ------------------------------------------------------------------ */
template <typename T>
void gerc_impl(int m, int n, std::complex<T> alpha, const std::complex<T>* x, int incx,
               const std::complex<T>* y, int incy, std::complex<T>* a, int lda) {
	using C = std::complex<T>;
	if (m <= 0 || n <= 0 || incx == 0 || incy == 0) {
		return;
	}
	if (alpha == C(0)) {
		return;
	}

	int jy = start_index(n, incy);
	for (int j = 0; j < n; ++j) {
		C temp = alpha * std::conj(y[jy]);
		int ix = start_index(m, incx);
		for (int i = 0; i < m; ++i) {
                    a[i + (j * lda)] += x[ix] * temp;
                    ix += incx;
		}
		jy += incy;
	}
}

/* ------------------------------------------------------------------ */
/* syr_impl — a ← α·x·xᵀ + a   (a symmetric)                       */
/* ------------------------------------------------------------------ */
template <typename T>
void syr_impl(char uplo, int n, T alpha, const T* x, int incx, T* a, int lda) {
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || incx == 0) {
		return;
	}
	if (alpha == T(0)) {
		return;
	}

	int jx = start_index(n, incx);
	if (ul == 'U') {
		for (int j = 0; j < n; ++j) {
			T temp = alpha * x[jx];
			int ix = start_index(n, incx);
			for (int i = 0; i <= j; ++i) {
                            a[i + (j * lda)] += x[ix] * temp;
                            ix += incx;
			}
			jx += incx;
		}
	} else {
		for (int j = 0; j < n; ++j) {
			T temp = alpha * x[jx];
			int ix = jx;
			for (int i = j; i < n; ++i) {
                            a[i + (j * lda)] += x[ix] * temp;
                            ix += incx;
			}
			jx += incx;
		}
	}
}

/* ------------------------------------------------------------------ */
/* her_impl — a ← α·x·xᴴ + a   (a Hermitian, α real)               */
/* ------------------------------------------------------------------ */
template <typename T>
void her_impl(char uplo, int n, T alpha, const std::complex<T>* x, int incx,
              std::complex<T>* a, int lda) {
	using C = std::complex<T>;
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || incx == 0) {
		return;
	}
	if (alpha == T(0)) {
		return;
	}

	int jx = start_index(n, incx);
	if (ul == 'U') {
		for (int j = 0; j < n; ++j) {
			C temp = C(alpha) * std::conj(x[jx]);
			int ix = start_index(n, incx);
			for (int i = 0; i < j; ++i) {
                            a[i + (j * lda)] += x[ix] * temp;
                            ix += incx;
			}
			// Diagonal is real
                        a[j + (j * lda)] = C(a[j + (j * lda)].real() + (x[jx] * temp).real());
                        jx += incx;
		}
	} else {
		for (int j = 0; j < n; ++j) {
			C temp = C(alpha) * std::conj(x[jx]);
			// Diagonal is real
                        a[j + (j * lda)] = C(a[j + (j * lda)].real() + (x[jx] * temp).real());
                        int ix = jx;
			for (int i = j + 1; i < n; ++i) {
				ix += incx;
                                a[i + (j * lda)] += x[ix] * temp;
                        }
			jx += incx;
		}
	}
}

/* ------------------------------------------------------------------ */
/* syr2_impl — a ← α·x·yᵀ + α·y·xᵀ + a   (a symmetric)            */
/* ------------------------------------------------------------------ */
template <typename T>
void syr2_impl(char uplo, int n, T alpha, const T* x, int incx, const T* y, int incy, T* a,
               int lda) {
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0) {
		return;
	}
	if (alpha == T(0)) {
		return;
	}

	int jx = start_index(n, incx);
	int jy = start_index(n, incy);
	if (ul == 'U') {
		for (int j = 0; j < n; ++j) {
			T temp1 = alpha * y[jy];
			T temp2 = alpha * x[jx];
			int ix = start_index(n, incx);
			int iy = start_index(n, incy);
			for (int i = 0; i <= j; ++i) {
                            a[i + (j * lda)] += (x[ix] * temp1) + (y[iy] * temp2);
                            ix += incx;
                            iy += incy;
			}
			jx += incx;
			jy += incy;
		}
	} else {
		for (int j = 0; j < n; ++j) {
			T temp1 = alpha * y[jy];
			T temp2 = alpha * x[jx];
			int ix = jx;
			int iy = jy;
			for (int i = j; i < n; ++i) {
                            a[i + (j * lda)] += (x[ix] * temp1) + (y[iy] * temp2);
                            ix += incx;
                            iy += incy;
			}
			jx += incx;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* her2_impl — a ← α·x·yᴴ + conj(α)·y·xᴴ + a   (a Hermitian)      */
/* ------------------------------------------------------------------ */
template <typename T>
void her2_impl(char uplo, int n, std::complex<T> alpha, const std::complex<T>* x, int incx,
               const std::complex<T>* y, int incy, std::complex<T>* a, int lda) {
	using C = std::complex<T>;
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0) {
		return;
	}
	if (alpha == C(0)) {
		return;
	}

	int jx = start_index(n, incx);
	int jy = start_index(n, incy);
	if (ul == 'U') {
		for (int j = 0; j < n; ++j) {
			C temp1 = alpha * std::conj(y[jy]);
			C temp2 = std::conj(alpha * x[jx]);
			int ix = start_index(n, incx);
			int iy = start_index(n, incy);
			for (int i = 0; i < j; ++i) {
                            a[i + (j * lda)] += (x[ix] * temp1) + (y[iy] * temp2);
                            ix += incx;
                            iy += incy;
			}
                        a[j + (j * lda)] =
                            C((a[j + (j * lda)] + (x[jx] * temp1) + (y[jy] * temp2)).real());
                        jx += incx;
			jy += incy;
		}
	} else {
		for (int j = 0; j < n; ++j) {
			C temp1 = alpha * std::conj(y[jy]);
			C temp2 = std::conj(alpha * x[jx]);
                        a[j + (j * lda)] =
                            C((a[j + (j * lda)] + (x[jx] * temp1) + (y[jy] * temp2)).real());
                        int ix = jx;
			int iy = jy;
			for (int i = j + 1; i < n; ++i) {
				ix += incx;
				iy += incy;
                                a[i + (j * lda)] += (x[ix] * temp1) + (y[iy] * temp2);
                        }
			jx += incx;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* gbmv_impl — y ← α·op(a)·x + β·y   (a banded)                    */
/* ------------------------------------------------------------------ */

} // anonymous namespace

void sgemv(char trans, int m, int n, float alpha, const float* a, int lda, const float* x,
           int incx, float beta, float* y, int incy) {
	gemv_impl(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
void dgemv(char trans, int m, int n, double alpha, const double* a, int lda, const double* x,
           int incx, double beta, double* y, int incy) {
	gemv_impl(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
void cgemv(char trans, int m, int n, std::complex<float> alpha, const std::complex<float>* a,
           int lda, const std::complex<float>* x, int incx, std::complex<float> beta,
           std::complex<float>* y, int incy) {
	gemv_impl(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
void zgemv(char trans, int m, int n, std::complex<double> alpha, const std::complex<double>* a,
           int lda, const std::complex<double>* x, int incx, std::complex<double> beta,
           std::complex<double>* y, int incy) {
	gemv_impl(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

/* symv */
void ssymv(char uplo, int n, float alpha, const float* a, int lda, const float* x, int incx,
           float beta, float* y, int incy) {
	symv_impl(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}
void dsymv(char uplo, int n, double alpha, const double* a, int lda, const double* x, int incx,
           double beta, double* y, int incy) {
	symv_impl(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

/* hemv */
void chemv(char uplo, int n, std::complex<float> alpha, const std::complex<float>* a, int lda,
           const std::complex<float>* x, int incx, std::complex<float> beta,
           std::complex<float>* y, int incy) {
	hemv_impl(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}
void zhemv(char uplo, int n, std::complex<double> alpha, const std::complex<double>* a, int lda,
           const std::complex<double>* x, int incx, std::complex<double> beta,
           std::complex<double>* y, int incy) {
	hemv_impl(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

/* trmv */
void strmv(char uplo, char trans, char diag, int n, const float* a, int lda, float* x,
           int incx) {
	trmv_impl(uplo, trans, diag, n, a, lda, x, incx);
}
void dtrmv(char uplo, char trans, char diag, int n, const double* a, int lda, double* x,
           int incx) {
	trmv_impl(uplo, trans, diag, n, a, lda, x, incx);
}
void ctrmv(char uplo, char trans, char diag, int n, const std::complex<float>* a, int lda,
           std::complex<float>* x, int incx) {
	trmv_impl(uplo, trans, diag, n, a, lda, x, incx);
}
void ztrmv(char uplo, char trans, char diag, int n, const std::complex<double>* a, int lda,
           std::complex<double>* x, int incx) {
	trmv_impl(uplo, trans, diag, n, a, lda, x, incx);
}

/* trsv */
void strsv(char uplo, char trans, char diag, int n, const float* a, int lda, float* x,
           int incx) {
	trsv_impl(uplo, trans, diag, n, a, lda, x, incx);
}
void dtrsv(char uplo, char trans, char diag, int n, const double* a, int lda, double* x,
           int incx) {
	trsv_impl(uplo, trans, diag, n, a, lda, x, incx);
}
void ctrsv(char uplo, char trans, char diag, int n, const std::complex<float>* a, int lda,
           std::complex<float>* x, int incx) {
	trsv_impl(uplo, trans, diag, n, a, lda, x, incx);
}
void ztrsv(char uplo, char trans, char diag, int n, const std::complex<double>* a, int lda,
           std::complex<double>* x, int incx) {
	trsv_impl(uplo, trans, diag, n, a, lda, x, incx);
}

/* ger */
void sger(int m, int n, float alpha, const float* x, int incx, const float* y, int incy,
          float* a, int lda) {
	ger_impl(m, n, alpha, x, incx, y, incy, a, lda);
}
void dger(int m, int n, double alpha, const double* x, int incx, const double* y, int incy,
          double* a, int lda) {
	ger_impl(m, n, alpha, x, incx, y, incy, a, lda);
}

/* geru / gerc */
void cgeru(int m, int n, std::complex<float> alpha, const std::complex<float>* x, int incx,
           const std::complex<float>* y, int incy, std::complex<float>* a, int lda) {
	geru_impl(m, n, alpha, x, incx, y, incy, a, lda);
}
void zgeru(int m, int n, std::complex<double> alpha, const std::complex<double>* x, int incx,
           const std::complex<double>* y, int incy, std::complex<double>* a, int lda) {
	geru_impl(m, n, alpha, x, incx, y, incy, a, lda);
}
void cgerc(int m, int n, std::complex<float> alpha, const std::complex<float>* x, int incx,
           const std::complex<float>* y, int incy, std::complex<float>* a, int lda) {
	gerc_impl(m, n, alpha, x, incx, y, incy, a, lda);
}
void zgerc(int m, int n, std::complex<double> alpha, const std::complex<double>* x, int incx,
           const std::complex<double>* y, int incy, std::complex<double>* a, int lda) {
	gerc_impl(m, n, alpha, x, incx, y, incy, a, lda);
}

/* syr */
void ssyr(char uplo, int n, float alpha, const float* x, int incx, float* a, int lda) {
	syr_impl(uplo, n, alpha, x, incx, a, lda);
}
void dsyr(char uplo, int n, double alpha, const double* x, int incx, double* a, int lda) {
	syr_impl(uplo, n, alpha, x, incx, a, lda);
}

/* her */
void cher(char uplo, int n, float alpha, const std::complex<float>* x, int incx,
          std::complex<float>* a, int lda) {
	her_impl(uplo, n, alpha, x, incx, a, lda);
}
void zher(char uplo, int n, double alpha, const std::complex<double>* x, int incx,
          std::complex<double>* a, int lda) {
	her_impl(uplo, n, alpha, x, incx, a, lda);
}

/* syr2 */
void ssyr2(char uplo, int n, float alpha, const float* x, int incx, const float* y, int incy,
           float* a, int lda) {
	syr2_impl(uplo, n, alpha, x, incx, y, incy, a, lda);
}
void dsyr2(char uplo, int n, double alpha, const double* x, int incx, const double* y, int incy,
           double* a, int lda) {
	syr2_impl(uplo, n, alpha, x, incx, y, incy, a, lda);
}

/* her2 */
void cher2(char uplo, int n, std::complex<float> alpha, const std::complex<float>* x, int incx,
           const std::complex<float>* y, int incy, std::complex<float>* a, int lda) {
	her2_impl(uplo, n, alpha, x, incx, y, incy, a, lda);
}
void zher2(char uplo, int n, std::complex<double> alpha, const std::complex<double>* x, int incx,
           const std::complex<double>* y, int incy, std::complex<double>* a, int lda) {
	her2_impl(uplo, n, alpha, x, incx, y, incy, a, lda);
}

/* gbmv */

} // namespace theblas
