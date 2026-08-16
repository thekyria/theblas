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

namespace {

template <typename T>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void gbmv_impl(char trans, int m, int n, int kl, int ku, T alpha, const T* a, int lda,
               const T* x, int incx, T beta, T* y, int incy) {
	char tr = to_upper(trans);
	if (!valid_trans(trans) || m <= 0 || n <= 0 || kl < 0 || ku < 0 || incx == 0 || incy == 0) {
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
		int jx = start_index(lenx, incx);
		for (int j = 0; j < n; ++j) {
			T temp = alpha * x[jx];
			int i_start = std::max(0, j - ku);
			int i_end = std::min(m - 1, j + kl);
                        int iy = start_index(leny, incy) + (i_start * incy);
                        for (int i = i_start; i <= i_end; ++i) {
                            y[iy] += temp * a[ku + i - j + (j * lda)];
                            iy += incy;
			}
			jx += incx;
		}
	} else if (tr == 'T') {
		int jy = start_index(leny, incy);
		for (int j = 0; j < n; ++j) {
			T temp = T(0);
			int i_start = std::max(0, j - ku);
			int i_end = std::min(m - 1, j + kl);
                        int ix = start_index(lenx, incx) + (i_start * incx);
                        for (int i = i_start; i <= i_end; ++i) {
                            temp += a[ku + i - j + (j * lda)] * x[ix];
                            ix += incx;
			}
			y[jy] += alpha * temp;
			jy += incy;
		}
	} else {
		// 'C' conjugate-transpose
		int jy = start_index(leny, incy);
		for (int j = 0; j < n; ++j) {
			T temp = T(0);
			int i_start = std::max(0, j - ku);
			int i_end = std::min(m - 1, j + kl);
                        int ix = start_index(lenx, incx) + (i_start * incx);
                        for (int i = i_start; i <= i_end; ++i) {
                            temp += conj_val(a[ku + i - j + (j * lda)]) * x[ix];
                            ix += incx;
			}
			y[jy] += alpha * temp;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* sbmv_impl — y ← α·a·x + β·y   (a symmetric banded)              */
/* ------------------------------------------------------------------ */
template <typename T>
void sbmv_impl(char uplo, int n, int k, T alpha, const T* a, int lda, const T* x, int incx,
               T beta, T* y, int incy) {
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || k < 0 || incx == 0 || incy == 0) {
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
			int l = k - j;
			int i_start = std::max(0, j - k);
                        int ix = start_index(n, incx) + (i_start * incx);
                        int iy = start_index(n, incy) + (i_start * incy);
                        for (int i = i_start; i < j; ++i) {
                            y[iy] += temp1 * a[l + i + (j * lda)];
                            temp2 += a[l + i + (j * lda)] * x[ix];
                            ix += incx;
                            iy += incy;
			}
                        y[jy] += (temp1 * a[k + (j * lda)]) + (alpha * temp2);
                        jx += incx;
			jy += incy;
		}
	} else {
		int jx = start_index(n, incx);
		int jy = start_index(n, incy);
		for (int j = 0; j < n; ++j) {
			T temp1 = alpha * x[jx];
			T temp2 = T(0);
                        y[jy] += temp1 * a[0 + (j * lda)];
                        int ix = jx;
			int iy = jy;
			int i_end = std::min(n - 1, j + k);
			for (int i = j + 1; i <= i_end; ++i) {
				ix += incx;
				iy += incy;
                                y[iy] += temp1 * a[i - j + (j * lda)];
                                temp2 += a[i - j + (j * lda)] * x[ix];
                        }
			y[jy] += alpha * temp2;
			jx += incx;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* hbmv_impl — y ← α·a·x + β·y   (a Hermitian banded)              */
/* ------------------------------------------------------------------ */
template <typename T>
void hbmv_impl(char uplo, int n, int k, std::complex<T> alpha, const std::complex<T>* a,
               int lda, const std::complex<T>* x, int incx, std::complex<T> beta,
               std::complex<T>* y, int incy) {
	using C = std::complex<T>;
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || k < 0 || incx == 0 || incy == 0) {
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
			int l = k - j;
			int i_start = std::max(0, j - k);
                        int ix = start_index(n, incx) + (i_start * incx);
                        int iy = start_index(n, incy) + (i_start * incy);
                        for (int i = i_start; i < j; ++i) {
                            y[iy] += temp1 * a[l + i + (j * lda)];
                            temp2 += std::conj(a[l + i + (j * lda)]) * x[ix];
                            ix += incx;
                            iy += incy;
			}
                        y[jy] += (temp1 * T(a[k + (j * lda)].real())) + (alpha * temp2);
                        jx += incx;
			jy += incy;
		}
	} else {
		int jx = start_index(n, incx);
		int jy = start_index(n, incy);
		for (int j = 0; j < n; ++j) {
			C temp1 = alpha * x[jx];
			C temp2 = C(0);
                        y[jy] += temp1 * T(a[0 + (j * lda)].real());
                        int ix = jx;
			int iy = jy;
			int i_end = std::min(n - 1, j + k);
			for (int i = j + 1; i <= i_end; ++i) {
				ix += incx;
				iy += incy;
                                y[iy] += temp1 * a[i - j + (j * lda)];
                                temp2 += std::conj(a[i - j + (j * lda)]) * x[ix];
                        }
			y[jy] += alpha * temp2;
			jx += incx;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* tbmv_impl — x ← op(a)·x   (a triangular banded)                  */
/* ------------------------------------------------------------------ */
template <typename T>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void tbmv_impl(char uplo, char trans, char diag, int n, int k, const T* a, int lda, T* x,
               int incx) {
	char ul = to_upper(uplo);
	char tr = to_upper(trans);
	char dg = to_upper(diag);
	if (!valid_uplo(uplo) || !valid_trans(trans) || !valid_diag(diag) || n <= 0 || k < 0 ||
	    incx == 0) {
		return;
	}
	bool unit = (dg == 'U');
	bool conj = (tr == 'C');

	if (tr == 'N') {
		if (ul == 'U') {
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				if (x[jx] != T(0)) {
					T temp = x[jx];
					int l = k - j;
					int i_start = std::max(0, j - k);
                                        int ix = start_index(n, incx) + (i_start * incx);
                                        for (int i = i_start; i < j; ++i) {
                                            x[ix] += temp * a[l + i + (j * lda)];
                                            ix += incx;
					}
					if (!unit) {
                                            x[jx] *= a[k + (j * lda)];
                                        }
				}
				jx += incx;
			}
		} else {
                    int jx = start_index(n, incx) + ((n - 1) * incx);
                    for (int j = n - 1; j >= 0; --j) {
                        if (x[jx] != T(0)) {
                            T temp = x[jx];
                            int i_end = std::min(n - 1, j + k);
                            int ix = start_index(n, incx) + (i_end * incx);
                            for (int i = i_end; i > j; --i) {
                                x[ix] += temp * a[i - j + (j * lda)];
                                ix -= incx;
                            }
                            if (!unit) {
                                x[jx] *= a[0 + (j * lda)];
                            }
                        }
                        jx -= incx;
                    }
		}
	} else {
		// Transpose or conjugate-transpose
		if (ul == 'U') {
                    int jx = start_index(n, incx) + ((n - 1) * incx);
                    for (int j = n - 1; j >= 0; --j) {
                        T temp = x[jx];
                        int l = k - j;
                        if (!unit) {
                            T aval = conj ? conj_val(a[k + (j * lda)]) : a[k + (j * lda)];
                            temp *= aval;
                        }
                        int i_start = std::max(0, j - k);
                        int ix = jx;
                        for (int i = j - 1; i >= i_start; --i) {
                            ix -= incx;
                            T aval = conj ? conj_val(a[l + i + (j * lda)]) : a[l + i + (j * lda)];
                            temp += aval * x[ix];
                        }
                        x[jx] = temp;
                        jx -= incx;
                    }
		} else {
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				T temp = x[jx];
				if (!unit) {
                                    T aval = conj ? conj_val(a[0 + (j * lda)]) : a[0 + (j * lda)];
                                    temp *= aval;
				}
				int i_end = std::min(n - 1, j + k);
				int ix = jx;
				for (int i = j + 1; i <= i_end; ++i) {
					ix += incx;
                                        T aval = conj ? conj_val(a[i - j + (j * lda)])
                                                      : a[i - j + (j * lda)];
                                        temp += aval * x[ix];
				}
				x[jx] = temp;
				jx += incx;
			}
		}
	}
}

/* ------------------------------------------------------------------ */
/* tbsv_impl — solve op(a)·x = b   (a triangular banded)             */
/* ------------------------------------------------------------------ */
template <typename T>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void tbsv_impl(char uplo, char trans, char diag, int n, int k, const T* a, int lda, T* x,
               int incx) {
	char ul = to_upper(uplo);
	char tr = to_upper(trans);
	char dg = to_upper(diag);
	if (!valid_uplo(uplo) || !valid_trans(trans) || !valid_diag(diag) || n <= 0 || k < 0 ||
	    incx == 0) {
		return;
	}
	bool unit = (dg == 'U');
	bool conj = (tr == 'C');

	if (tr == 'N') {
		if (ul == 'U') {
                    int jx = start_index(n, incx) + ((n - 1) * incx);
                    for (int j = n - 1; j >= 0; --j) {
                        if (!unit) {
                            x[jx] /= a[k + (j * lda)];
                        }
                        T temp = x[jx];
                        int l = k - j;
                        int i_start = std::max(0, j - k);
                        int ix = jx;
                        for (int i = j - 1; i >= i_start; --i) {
                            ix -= incx;
                            x[ix] -= temp * a[l + i + (j * lda)];
                        }
                        jx -= incx;
                    }
		} else {
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				if (!unit) {
                                    x[jx] /= a[0 + (j * lda)];
                                }
				T temp = x[jx];
				int i_end = std::min(n - 1, j + k);
				int ix = jx;
				for (int i = j + 1; i <= i_end; ++i) {
					ix += incx;
                                        x[ix] -= temp * a[i - j + (j * lda)];
                                }
				jx += incx;
			}
		}
	} else {
		if (ul == 'U') {
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				T temp = x[jx];
				int l = k - j;
				int i_start = std::max(0, j - k);
                                int ix = start_index(n, incx) + (i_start * incx);
                                for (int i = i_start; i < j; ++i) {
                                    T aval = conj ? conj_val(a[l + i + (j * lda)])
                                                  : a[l + i + (j * lda)];
                                    temp -= aval * x[ix];
                                    ix += incx;
				}
				if (!unit) {
                                    T aval = conj ? conj_val(a[k + (j * lda)]) : a[k + (j * lda)];
                                    temp /= aval;
				}
				x[jx] = temp;
				jx += incx;
			}
		} else {
                    int jx = start_index(n, incx) + ((n - 1) * incx);
                    for (int j = n - 1; j >= 0; --j) {
                        T temp = x[jx];
                        int i_end = std::min(n - 1, j + k);
                        int ix = start_index(n, incx) + (i_end * incx);
                        for (int i = i_end; i > j; --i) {
                            T aval = conj ? conj_val(a[i - j + (j * lda)]) : a[i - j + (j * lda)];
                            temp -= aval * x[ix];
                            ix -= incx;
                        }
                        if (!unit) {
                            T aval = conj ? conj_val(a[0 + (j * lda)]) : a[0 + (j * lda)];
                            temp /= aval;
                        }
                        x[jx] = temp;
                        jx -= incx;
                    }
		}
	}
}

/* ------------------------------------------------------------------ */
/* spmv_impl — y ← α·a·x + β·y   (a symmetric, packed)             */
/* ------------------------------------------------------------------ */

} // anonymous namespace

void sgbmv(char trans, int m, int n, int kl, int ku, float alpha, const float* a, int lda,
           const float* x, int incx, float beta, float* y, int incy) {
	gbmv_impl(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}
void dgbmv(char trans, int m, int n, int kl, int ku, double alpha, const double* a, int lda,
           const double* x, int incx, double beta, double* y, int incy) {
	gbmv_impl(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}
void cgbmv(char trans, int m, int n, int kl, int ku, std::complex<float> alpha,
           const std::complex<float>* a, int lda, const std::complex<float>* x, int incx,
           std::complex<float> beta, std::complex<float>* y, int incy) {
	gbmv_impl(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}
void zgbmv(char trans, int m, int n, int kl, int ku, std::complex<double> alpha,
           const std::complex<double>* a, int lda, const std::complex<double>* x, int incx,
           std::complex<double> beta, std::complex<double>* y, int incy) {
	gbmv_impl(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

/* sbmv */
void ssbmv(char uplo, int n, int k, float alpha, const float* a, int lda, const float* x,
           int incx, float beta, float* y, int incy) {
	sbmv_impl(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
void dsbmv(char uplo, int n, int k, double alpha, const double* a, int lda, const double* x,
           int incx, double beta, double* y, int incy) {
	sbmv_impl(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

/* hbmv */
void chbmv(char uplo, int n, int k, std::complex<float> alpha, const std::complex<float>* a,
           int lda, const std::complex<float>* x, int incx, std::complex<float> beta,
           std::complex<float>* y, int incy) {
	hbmv_impl(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
void zhbmv(char uplo, int n, int k, std::complex<double> alpha, const std::complex<double>* a,
           int lda, const std::complex<double>* x, int incx, std::complex<double> beta,
           std::complex<double>* y, int incy) {
	hbmv_impl(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

/* tbmv */
void stbmv(char uplo, char trans, char diag, int n, int k, const float* a, int lda, float* x,
           int incx) {
	tbmv_impl(uplo, trans, diag, n, k, a, lda, x, incx);
}
void dtbmv(char uplo, char trans, char diag, int n, int k, const double* a, int lda, double* x,
           int incx) {
	tbmv_impl(uplo, trans, diag, n, k, a, lda, x, incx);
}
void ctbmv(char uplo, char trans, char diag, int n, int k, const std::complex<float>* a, int lda,
           std::complex<float>* x, int incx) {
	tbmv_impl(uplo, trans, diag, n, k, a, lda, x, incx);
}
void ztbmv(char uplo, char trans, char diag, int n, int k, const std::complex<double>* a,
           int lda, std::complex<double>* x, int incx) {
	tbmv_impl(uplo, trans, diag, n, k, a, lda, x, incx);
}

/* tbsv */
void stbsv(char uplo, char trans, char diag, int n, int k, const float* a, int lda, float* x,
           int incx) {
	tbsv_impl(uplo, trans, diag, n, k, a, lda, x, incx);
}
void dtbsv(char uplo, char trans, char diag, int n, int k, const double* a, int lda, double* x,
           int incx) {
	tbsv_impl(uplo, trans, diag, n, k, a, lda, x, incx);
}
void ctbsv(char uplo, char trans, char diag, int n, int k, const std::complex<float>* a, int lda,
           std::complex<float>* x, int incx) {
	tbsv_impl(uplo, trans, diag, n, k, a, lda, x, incx);
}
void ztbsv(char uplo, char trans, char diag, int n, int k, const std::complex<double>* a,
           int lda, std::complex<double>* x, int incx) {
	tbsv_impl(uplo, trans, diag, n, k, a, lda, x, incx);
}

/* spmv */

} // namespace theblas
