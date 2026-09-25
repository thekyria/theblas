#include "theblas/theblas.h"

#include "theblas_detail.hpp"

#include <algorithm>
#include <complex>

namespace theblas {

using detail::conj_val;
using detail::idx;
using detail::index_t;
using detail::offset;
using detail::report_error;
using detail::start_index;
using detail::to_upper;
using detail::valid_diag;
using detail::valid_trans;
using detail::valid_uplo;

namespace {

int gbmv_info(char trans, int m, int n, int kl, int ku, int lda, int incx, int incy) {
    if (!valid_trans(trans)) {
        return 1;
    }
    if (m < 0) {
        return 2;
    }
    if (n < 0) {
        return 3;
    }
    if (kl < 0) {
        return 4;
    }
    if (ku < 0) {
        return 5;
    }
    if (lda < kl + ku + 1) {
        return 8;
    }
    if (incx == 0) {
        return 10;
    }
    if (incy == 0) {
        return 13;
    }
    return 0;
}

int sbmv_info(char uplo, int n, int k, int lda, int incx, int incy) {
    if (!valid_uplo(uplo)) {
        return 1;
    }
    if (n < 0) {
        return 2;
    }
    if (k < 0) {
        return 3;
    }
    if (lda < k + 1) {
        return 6;
    }
    if (incx == 0) {
        return 8;
    }
    if (incy == 0) {
        return 11;
    }
    return 0;
}

int tbmv_info(char uplo, char trans, char diag, int n, int k, int lda, int incx) {
    if (!valid_uplo(uplo)) {
        return 1;
    }
    if (!valid_trans(trans)) {
        return 2;
    }
    if (!valid_diag(diag)) {
        return 3;
    }
    if (n < 0) {
        return 4;
    }
    if (k < 0) {
        return 5;
    }
    if (lda < k + 1) {
        return 7;
    }
    if (incx == 0) {
        return 9;
    }
    return 0;
}

template <typename T>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void gbmv_impl(const char *routine, char trans, int m, int n, int kl, int ku, T alpha, const T *a,
               int lda, const T *x, int incx, T beta, T *y, int incy) {
    const int info = gbmv_info(trans, m, n, kl, ku, lda, incx, incy);
    if (info != 0) {
        report_error(routine, info);
        return;
    }
    if (m == 0 || n == 0) {
        return;
    }
    char tr = to_upper(trans);
    int leny = (tr == 'N') ? m : n;
    int lenx = (tr == 'N') ? n : m;

    // Scale y by beta
    {
        index_t iy = start_index(leny, incy);
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
        index_t jx = start_index(lenx, incx);
        for (int j = 0; j < n; ++j) {
            T temp = alpha * x[jx];
            int i_start = std::max(0, j - ku);
            int i_end = std::min(m - 1, j + kl);
            index_t iy = start_index(leny, incy) + offset(i_start, incy);
            for (int i = i_start; i <= i_end; ++i) {
                y[iy] += temp * a[idx(ku + i - j, j, lda)];
                iy += incy;
            }
            jx += incx;
        }
    } else if (tr == 'T') {
        index_t jy = start_index(leny, incy);
        for (int j = 0; j < n; ++j) {
            T temp = T(0);
            int i_start = std::max(0, j - ku);
            int i_end = std::min(m - 1, j + kl);
            index_t ix = start_index(lenx, incx) + offset(i_start, incx);
            for (int i = i_start; i <= i_end; ++i) {
                temp += a[idx(ku + i - j, j, lda)] * x[ix];
                ix += incx;
            }
            y[jy] += alpha * temp;
            jy += incy;
        }
    } else {
        // 'C' conjugate-transpose
        index_t jy = start_index(leny, incy);
        for (int j = 0; j < n; ++j) {
            T temp = T(0);
            int i_start = std::max(0, j - ku);
            int i_end = std::min(m - 1, j + kl);
            index_t ix = start_index(lenx, incx) + offset(i_start, incx);
            for (int i = i_start; i <= i_end; ++i) {
                temp += conj_val(a[idx(ku + i - j, j, lda)]) * x[ix];
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
void sbmv_impl(const char *routine, char uplo, int n, int k, T alpha, const T *a, int lda,
               const T *x, int incx, T beta, T *y, int incy) {
    const int info = sbmv_info(uplo, n, k, lda, incx, incy);
    if (info != 0) {
        report_error(routine, info);
        return;
    }
    if (n == 0) {
        return;
    }
    char ul = to_upper(uplo);

    // Scale y by beta
    {
        index_t iy = start_index(n, incy);
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
        index_t jx = start_index(n, incx);
        index_t jy = start_index(n, incy);
        for (int j = 0; j < n; ++j) {
            T temp1 = alpha * x[jx];
            T temp2 = T(0);
            int l = k - j;
            int i_start = std::max(0, j - k);
            index_t ix = start_index(n, incx) + offset(i_start, incx);
            index_t iy = start_index(n, incy) + offset(i_start, incy);
            for (int i = i_start; i < j; ++i) {
                y[iy] += temp1 * a[idx(l + i, j, lda)];
                temp2 += a[idx(l + i, j, lda)] * x[ix];
                ix += incx;
                iy += incy;
            }
            y[jy] += (temp1 * a[idx(k, j, lda)]) + (alpha * temp2);
            jx += incx;
            jy += incy;
        }
    } else {
        index_t jx = start_index(n, incx);
        index_t jy = start_index(n, incy);
        for (int j = 0; j < n; ++j) {
            T temp1 = alpha * x[jx];
            T temp2 = T(0);
            y[jy] += temp1 * a[idx(0, j, lda)];
            index_t ix = jx;
            index_t iy = jy;
            int i_end = std::min(n - 1, j + k);
            for (int i = j + 1; i <= i_end; ++i) {
                ix += incx;
                iy += incy;
                y[iy] += temp1 * a[idx(i - j, j, lda)];
                temp2 += a[idx(i - j, j, lda)] * x[ix];
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
void hbmv_impl(const char *routine, char uplo, int n, int k, std::complex<T> alpha,
               const std::complex<T> *a, int lda, const std::complex<T> *x, int incx,
               std::complex<T> beta, std::complex<T> *y, int incy) {
    using C = std::complex<T>;
    const int info = sbmv_info(uplo, n, k, lda, incx, incy);
    if (info != 0) {
        report_error(routine, info);
        return;
    }
    if (n == 0) {
        return;
    }
    char ul = to_upper(uplo);

    // Scale y by beta
    {
        index_t iy = start_index(n, incy);
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
        index_t jx = start_index(n, incx);
        index_t jy = start_index(n, incy);
        for (int j = 0; j < n; ++j) {
            C temp1 = alpha * x[jx];
            C temp2 = C(0);
            int l = k - j;
            int i_start = std::max(0, j - k);
            index_t ix = start_index(n, incx) + offset(i_start, incx);
            index_t iy = start_index(n, incy) + offset(i_start, incy);
            for (int i = i_start; i < j; ++i) {
                y[iy] += temp1 * a[idx(l + i, j, lda)];
                temp2 += std::conj(a[idx(l + i, j, lda)]) * x[ix];
                ix += incx;
                iy += incy;
            }
            y[jy] += (temp1 * T(a[idx(k, j, lda)].real())) + (alpha * temp2);
            jx += incx;
            jy += incy;
        }
    } else {
        index_t jx = start_index(n, incx);
        index_t jy = start_index(n, incy);
        for (int j = 0; j < n; ++j) {
            C temp1 = alpha * x[jx];
            C temp2 = C(0);
            y[jy] += temp1 * T(a[idx(0, j, lda)].real());
            index_t ix = jx;
            index_t iy = jy;
            int i_end = std::min(n - 1, j + k);
            for (int i = j + 1; i <= i_end; ++i) {
                ix += incx;
                iy += incy;
                y[iy] += temp1 * a[idx(i - j, j, lda)];
                temp2 += std::conj(a[idx(i - j, j, lda)]) * x[ix];
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
void tbmv_impl(const char *routine, char uplo, char trans, char diag, int n, int k, const T *a,
               int lda, T *x, int incx) {
    const int info = tbmv_info(uplo, trans, diag, n, k, lda, incx);
    if (info != 0) {
        report_error(routine, info);
        return;
    }
    if (n == 0) {
        return;
    }
    char ul = to_upper(uplo);
    char tr = to_upper(trans);
    char dg = to_upper(diag);
    bool unit = (dg == 'U');
    bool conj = (tr == 'C');

    if (tr == 'N') {
        if (ul == 'U') {
            index_t jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                if (x[jx] != T(0)) {
                    T temp = x[jx];
                    int l = k - j;
                    int i_start = std::max(0, j - k);
                    index_t ix = start_index(n, incx) + offset(i_start, incx);
                    for (int i = i_start; i < j; ++i) {
                        x[ix] += temp * a[idx(l + i, j, lda)];
                        ix += incx;
                    }
                    if (!unit) {
                        x[jx] *= a[idx(k, j, lda)];
                    }
                }
                jx += incx;
            }
        } else {
            index_t jx = start_index(n, incx) + offset(n - 1, incx);
            for (int j = n - 1; j >= 0; --j) {
                if (x[jx] != T(0)) {
                    T temp = x[jx];
                    int i_end = std::min(n - 1, j + k);
                    index_t ix = start_index(n, incx) + offset(i_end, incx);
                    for (int i = i_end; i > j; --i) {
                        x[ix] += temp * a[idx(i - j, j, lda)];
                        ix -= incx;
                    }
                    if (!unit) {
                        x[jx] *= a[idx(0, j, lda)];
                    }
                }
                jx -= incx;
            }
        }
    } else {
        // Transpose or conjugate-transpose
        if (ul == 'U') {
            index_t jx = start_index(n, incx) + offset(n - 1, incx);
            for (int j = n - 1; j >= 0; --j) {
                T temp = x[jx];
                int l = k - j;
                if (!unit) {
                    T aval = conj ? conj_val(a[idx(k, j, lda)]) : a[idx(k, j, lda)];
                    temp *= aval;
                }
                int i_start = std::max(0, j - k);
                index_t ix = jx;
                for (int i = j - 1; i >= i_start; --i) {
                    ix -= incx;
                    T aval = conj ? conj_val(a[idx(l + i, j, lda)]) : a[idx(l + i, j, lda)];
                    temp += aval * x[ix];
                }
                x[jx] = temp;
                jx -= incx;
            }
        } else {
            index_t jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                T temp = x[jx];
                if (!unit) {
                    T aval = conj ? conj_val(a[idx(0, j, lda)]) : a[idx(0, j, lda)];
                    temp *= aval;
                }
                int i_end = std::min(n - 1, j + k);
                index_t ix = jx;
                for (int i = j + 1; i <= i_end; ++i) {
                    ix += incx;
                    T aval = conj ? conj_val(a[idx(i - j, j, lda)]) : a[idx(i - j, j, lda)];
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
void tbsv_impl(const char *routine, char uplo, char trans, char diag, int n, int k, const T *a,
               int lda, T *x, int incx) {
    const int info = tbmv_info(uplo, trans, diag, n, k, lda, incx);
    if (info != 0) {
        report_error(routine, info);
        return;
    }
    if (n == 0) {
        return;
    }
    char ul = to_upper(uplo);
    char tr = to_upper(trans);
    char dg = to_upper(diag);
    bool unit = (dg == 'U');
    bool conj = (tr == 'C');

    if (tr == 'N') {
        if (ul == 'U') {
            index_t jx = start_index(n, incx) + offset(n - 1, incx);
            for (int j = n - 1; j >= 0; --j) {
                if (!unit) {
                    x[jx] /= a[idx(k, j, lda)];
                }
                T temp = x[jx];
                int l = k - j;
                int i_start = std::max(0, j - k);
                index_t ix = jx;
                for (int i = j - 1; i >= i_start; --i) {
                    ix -= incx;
                    x[ix] -= temp * a[idx(l + i, j, lda)];
                }
                jx -= incx;
            }
        } else {
            index_t jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                if (!unit) {
                    x[jx] /= a[idx(0, j, lda)];
                }
                T temp = x[jx];
                int i_end = std::min(n - 1, j + k);
                index_t ix = jx;
                for (int i = j + 1; i <= i_end; ++i) {
                    ix += incx;
                    x[ix] -= temp * a[idx(i - j, j, lda)];
                }
                jx += incx;
            }
        }
    } else {
        if (ul == 'U') {
            index_t jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                T temp = x[jx];
                int l = k - j;
                int i_start = std::max(0, j - k);
                index_t ix = start_index(n, incx) + offset(i_start, incx);
                for (int i = i_start; i < j; ++i) {
                    T aval = conj ? conj_val(a[idx(l + i, j, lda)]) : a[idx(l + i, j, lda)];
                    temp -= aval * x[ix];
                    ix += incx;
                }
                if (!unit) {
                    T aval = conj ? conj_val(a[idx(k, j, lda)]) : a[idx(k, j, lda)];
                    temp /= aval;
                }
                x[jx] = temp;
                jx += incx;
            }
        } else {
            index_t jx = start_index(n, incx) + offset(n - 1, incx);
            for (int j = n - 1; j >= 0; --j) {
                T temp = x[jx];
                int i_end = std::min(n - 1, j + k);
                index_t ix = start_index(n, incx) + offset(i_end, incx);
                for (int i = i_end; i > j; --i) {
                    T aval = conj ? conj_val(a[idx(i - j, j, lda)]) : a[idx(i - j, j, lda)];
                    temp -= aval * x[ix];
                    ix -= incx;
                }
                if (!unit) {
                    T aval = conj ? conj_val(a[idx(0, j, lda)]) : a[idx(0, j, lda)];
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

void sgbmv(char trans, int m, int n, int kl, int ku, float alpha, const float *a, int lda,
           const float *x, int incx, float beta, float *y, int incy) {
    gbmv_impl("SGBMV", trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}
void dgbmv(char trans, int m, int n, int kl, int ku, double alpha, const double *a, int lda,
           const double *x, int incx, double beta, double *y, int incy) {
    gbmv_impl("DGBMV", trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}
void cgbmv(char trans, int m, int n, int kl, int ku, std::complex<float> alpha,
           const std::complex<float> *a, int lda, const std::complex<float> *x, int incx,
           std::complex<float> beta, std::complex<float> *y, int incy) {
    gbmv_impl("CGBMV", trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}
void zgbmv(char trans, int m, int n, int kl, int ku, std::complex<double> alpha,
           const std::complex<double> *a, int lda, const std::complex<double> *x, int incx,
           std::complex<double> beta, std::complex<double> *y, int incy) {
    gbmv_impl("ZGBMV", trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

/* sbmv */
void ssbmv(char uplo, int n, int k, float alpha, const float *a, int lda, const float *x, int incx,
           float beta, float *y, int incy) {
    sbmv_impl("SSBMV", uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
void dsbmv(char uplo, int n, int k, double alpha, const double *a, int lda, const double *x,
           int incx, double beta, double *y, int incy) {
    sbmv_impl("DSBMV", uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

/* hbmv */
void chbmv(char uplo, int n, int k, std::complex<float> alpha, const std::complex<float> *a,
           int lda, const std::complex<float> *x, int incx, std::complex<float> beta,
           std::complex<float> *y, int incy) {
    hbmv_impl("CHBMV", uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
void zhbmv(char uplo, int n, int k, std::complex<double> alpha, const std::complex<double> *a,
           int lda, const std::complex<double> *x, int incx, std::complex<double> beta,
           std::complex<double> *y, int incy) {
    hbmv_impl("ZHBMV", uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

/* tbmv */
void stbmv(char uplo, char trans, char diag, int n, int k, const float *a, int lda, float *x,
           int incx) {
    tbmv_impl("STBMV", uplo, trans, diag, n, k, a, lda, x, incx);
}
void dtbmv(char uplo, char trans, char diag, int n, int k, const double *a, int lda, double *x,
           int incx) {
    tbmv_impl("DTBMV", uplo, trans, diag, n, k, a, lda, x, incx);
}
void ctbmv(char uplo, char trans, char diag, int n, int k, const std::complex<float> *a, int lda,
           std::complex<float> *x, int incx) {
    tbmv_impl("CTBMV", uplo, trans, diag, n, k, a, lda, x, incx);
}
void ztbmv(char uplo, char trans, char diag, int n, int k, const std::complex<double> *a, int lda,
           std::complex<double> *x, int incx) {
    tbmv_impl("ZTBMV", uplo, trans, diag, n, k, a, lda, x, incx);
}

/* tbsv */
void stbsv(char uplo, char trans, char diag, int n, int k, const float *a, int lda, float *x,
           int incx) {
    tbsv_impl("STBSV", uplo, trans, diag, n, k, a, lda, x, incx);
}
void dtbsv(char uplo, char trans, char diag, int n, int k, const double *a, int lda, double *x,
           int incx) {
    tbsv_impl("DTBSV", uplo, trans, diag, n, k, a, lda, x, incx);
}
void ctbsv(char uplo, char trans, char diag, int n, int k, const std::complex<float> *a, int lda,
           std::complex<float> *x, int incx) {
    tbsv_impl("CTBSV", uplo, trans, diag, n, k, a, lda, x, incx);
}
void ztbsv(char uplo, char trans, char diag, int n, int k, const std::complex<double> *a, int lda,
           std::complex<double> *x, int incx) {
    tbsv_impl("ZTBSV", uplo, trans, diag, n, k, a, lda, x, incx);
}

/* spmv */

} // namespace theblas
