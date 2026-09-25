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

/* ====================================================================
 * Level 2 BLAS — Matrix-Vector Operations
 * ==================================================================== */

namespace {

int gemv_info(char trans, int m, int n, int lda, int incx, int incy) {
    if (!valid_trans(trans)) {
        return 1;
    }
    if (m < 0) {
        return 2;
    }
    if (n < 0) {
        return 3;
    }
    if (lda < std::max(1, m)) {
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

int symv_info(char uplo, int n, int lda, int incx, int incy) {
    if (!valid_uplo(uplo)) {
        return 1;
    }
    if (n < 0) {
        return 2;
    }
    if (lda < std::max(1, n)) {
        return 5;
    }
    if (incx == 0) {
        return 7;
    }
    if (incy == 0) {
        return 10;
    }
    return 0;
}

int trmv_info(char uplo, char trans, char diag, int n, int lda, int incx) {
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
    if (lda < std::max(1, n)) {
        return 6;
    }
    if (incx == 0) {
        return 8;
    }
    return 0;
}

int ger_info(int m, int n, int incx, int incy, int lda) {
    if (m < 0) {
        return 1;
    }
    if (n < 0) {
        return 2;
    }
    if (incx == 0) {
        return 5;
    }
    if (incy == 0) {
        return 7;
    }
    if (lda < std::max(1, m)) {
        return 9;
    }
    return 0;
}

int syr_info(char uplo, int n, int incx, int lda) {
    if (!valid_uplo(uplo)) {
        return 1;
    }
    if (n < 0) {
        return 2;
    }
    if (incx == 0) {
        return 5;
    }
    if (lda < std::max(1, n)) {
        return 7;
    }
    return 0;
}

int syr2_info(char uplo, int n, int incx, int incy, int lda) {
    if (!valid_uplo(uplo)) {
        return 1;
    }
    if (n < 0) {
        return 2;
    }
    if (incx == 0) {
        return 5;
    }
    if (incy == 0) {
        return 7;
    }
    if (lda < std::max(1, n)) {
        return 9;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* gemv_impl — y ← α·op(a)·x + β·y                                  */
/* ------------------------------------------------------------------ */
template <typename T>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void gemv_impl(const char *routine, char trans, int m, int n, T alpha, const T *a, int lda,
               const T *x, int incx, T beta, T *y, int incy) {
    const int info = gemv_info(trans, m, n, lda, incx, incy);
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
        // y ← α·a·x + y
        index_t jx = start_index(lenx, incx);
        for (int j = 0; j < n; ++j) {
            T temp = alpha * x[jx];
            index_t iy = start_index(leny, incy);
            for (int i = 0; i < m; ++i) {
                y[iy] += temp * a[idx(i, j, lda)];
                iy += incy;
            }
            jx += incx;
        }
    } else if (tr == 'T') {
        // y ← α·Aᵀ·x + y
        index_t jy = start_index(leny, incy);
        for (int j = 0; j < n; ++j) {
            T temp = T(0);
            index_t ix = start_index(lenx, incx);
            for (int i = 0; i < m; ++i) {
                temp += a[idx(i, j, lda)] * x[ix];
                ix += incx;
            }
            y[jy] += alpha * temp;
            jy += incy;
        }
    } else {
        // tr == 'C': y ← α·Aᴴ·x + y (conjugate-transpose)
        index_t jy = start_index(leny, incy);
        for (int j = 0; j < n; ++j) {
            T temp = T(0);
            index_t ix = start_index(lenx, incx);
            for (int i = 0; i < m; ++i) {
                temp += conj_val(a[idx(i, j, lda)]) * x[ix];
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
void symv_impl(const char *routine, char uplo, int n, T alpha, const T *a, int lda, const T *x,
               int incx, T beta, T *y, int incy) {
    const int info = symv_info(uplo, n, lda, incx, incy);
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
            index_t ix = start_index(n, incx);
            index_t iy = start_index(n, incy);
            for (int i = 0; i < j; ++i) {
                y[iy] += temp1 * a[idx(i, j, lda)];
                temp2 += a[idx(i, j, lda)] * x[ix];
                ix += incx;
                iy += incy;
            }
            y[jy] += (temp1 * a[idx(j, j, lda)]) + (alpha * temp2);
            jx += incx;
            jy += incy;
        }
    } else {
        index_t jx = start_index(n, incx);
        index_t jy = start_index(n, incy);
        for (int j = 0; j < n; ++j) {
            T temp1 = alpha * x[jx];
            T temp2 = T(0);
            y[jy] += temp1 * a[idx(j, j, lda)];
            index_t ix = jx;
            index_t iy = jy;
            for (int i = j + 1; i < n; ++i) {
                ix += incx;
                iy += incy;
                y[iy] += temp1 * a[idx(i, j, lda)];
                temp2 += a[idx(i, j, lda)] * x[ix];
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
void hemv_impl(const char *routine, char uplo, int n, std::complex<T> alpha,
               const std::complex<T> *a, int lda, const std::complex<T> *x, int incx,
               std::complex<T> beta, std::complex<T> *y, int incy) {
    using C = std::complex<T>;
    const int info = symv_info(uplo, n, lda, incx, incy);
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
            index_t ix = start_index(n, incx);
            index_t iy = start_index(n, incy);
            for (int i = 0; i < j; ++i) {
                y[iy] += temp1 * a[idx(i, j, lda)];
                temp2 += std::conj(a[idx(i, j, lda)]) * x[ix];
                ix += incx;
                iy += incy;
            }
            // Diagonal of a Hermitian matrix is real
            y[jy] += (temp1 * T(a[idx(j, j, lda)].real())) + (alpha * temp2);
            jx += incx;
            jy += incy;
        }
    } else {
        index_t jx = start_index(n, incx);
        index_t jy = start_index(n, incy);
        for (int j = 0; j < n; ++j) {
            C temp1 = alpha * x[jx];
            C temp2 = C(0);
            y[jy] += temp1 * T(a[idx(j, j, lda)].real());
            index_t ix = jx;
            index_t iy = jy;
            for (int i = j + 1; i < n; ++i) {
                ix += incx;
                iy += incy;
                y[iy] += temp1 * a[idx(i, j, lda)];
                temp2 += std::conj(a[idx(i, j, lda)]) * x[ix];
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
void trmv_impl(const char *routine, char uplo, char trans, char diag, int n, const T *a, int lda,
               T *x, int incx) {
    const int info = trmv_info(uplo, trans, diag, n, lda, incx);
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

    if (tr == 'N') {
        if (ul == 'U') {
            index_t jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                if (x[jx] != T(0)) {
                    T temp = x[jx];
                    index_t ix = start_index(n, incx);
                    for (int i = 0; i < j; ++i) {
                        x[ix] += temp * a[idx(i, j, lda)];
                        ix += incx;
                    }
                    if (!unit) {
                        x[jx] *= a[idx(j, j, lda)];
                    }
                }
                jx += incx;
            }
        } else {
            index_t jx = start_index(n, incx) + offset(n - 1, incx);
            for (int j = n - 1; j >= 0; --j) {
                if (x[jx] != T(0)) {
                    T temp = x[jx];
                    index_t ix = start_index(n, incx) + offset(n - 1, incx);
                    for (int i = n - 1; i > j; --i) {
                        x[ix] += temp * a[idx(i, j, lda)];
                        ix -= incx;
                    }
                    if (!unit) {
                        x[jx] *= a[idx(j, j, lda)];
                    }
                }
                jx -= incx;
            }
        }
    } else {
        // Transpose or conjugate-transpose
        bool conj = (tr == 'C');
        if (ul == 'U') {
            index_t jx = start_index(n, incx) + offset(n - 1, incx);
            for (int j = n - 1; j >= 0; --j) {
                T temp = x[jx];
                if (!unit) {
                    temp *= conj ? conj_val(a[idx(j, j, lda)]) : a[idx(j, j, lda)];
                }
                index_t ix = jx;
                for (int i = j - 1; i >= 0; --i) {
                    ix -= incx;
                    temp += (conj ? conj_val(a[idx(i, j, lda)]) : a[idx(i, j, lda)]) * x[ix];
                }
                x[jx] = temp;
                jx -= incx;
            }
        } else {
            index_t jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                T temp = x[jx];
                if (!unit) {
                    temp *= conj ? conj_val(a[idx(j, j, lda)]) : a[idx(j, j, lda)];
                }
                index_t ix = jx;
                for (int i = j + 1; i < n; ++i) {
                    ix += incx;
                    temp += (conj ? conj_val(a[idx(i, j, lda)]) : a[idx(i, j, lda)]) * x[ix];
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
void trsv_impl(const char *routine, char uplo, char trans, char diag, int n, const T *a, int lda,
               T *x, int incx) {
    const int info = trmv_info(uplo, trans, diag, n, lda, incx);
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

    if (tr == 'N') {
        if (ul == 'U') {
            // Back-substitution
            index_t jx = start_index(n, incx) + offset(n - 1, incx);
            for (int j = n - 1; j >= 0; --j) {
                if (!unit) {
                    x[jx] /= a[idx(j, j, lda)];
                }
                T temp = x[jx];
                index_t ix = jx;
                for (int i = j - 1; i >= 0; --i) {
                    ix -= incx;
                    x[ix] -= temp * a[idx(i, j, lda)];
                }
                jx -= incx;
            }
        } else {
            // Forward-substitution
            index_t jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                if (!unit) {
                    x[jx] /= a[idx(j, j, lda)];
                }
                T temp = x[jx];
                index_t ix = jx;
                for (int i = j + 1; i < n; ++i) {
                    ix += incx;
                    x[ix] -= temp * a[idx(i, j, lda)];
                }
                jx += incx;
            }
        }
    } else {
        bool conj = (tr == 'C');
        if (ul == 'U') {
            index_t jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                T temp = x[jx];
                index_t ix = start_index(n, incx);
                for (int i = 0; i < j; ++i) {
                    temp -= (conj ? conj_val(a[idx(i, j, lda)]) : a[idx(i, j, lda)]) * x[ix];
                    ix += incx;
                }
                if (!unit) {
                    temp /= conj ? conj_val(a[idx(j, j, lda)]) : a[idx(j, j, lda)];
                }
                x[jx] = temp;
                jx += incx;
            }
        } else {
            index_t jx = start_index(n, incx) + offset(n - 1, incx);
            for (int j = n - 1; j >= 0; --j) {
                T temp = x[jx];
                index_t ix = start_index(n, incx) + offset(n - 1, incx);
                for (int i = n - 1; i > j; --i) {
                    temp -= (conj ? conj_val(a[idx(i, j, lda)]) : a[idx(i, j, lda)]) * x[ix];
                    ix -= incx;
                }
                if (!unit) {
                    temp /= conj ? conj_val(a[idx(j, j, lda)]) : a[idx(j, j, lda)];
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
void ger_impl(const char *routine, int m, int n, T alpha, const T *x, int incx, const T *y,
              int incy, T *a, int lda) {
    const int info = ger_info(m, n, incx, incy, lda);
    if (info != 0) {
        report_error(routine, info);
        return;
    }
    if (m == 0 || n == 0) {
        return;
    }
    if (alpha == T(0)) {
        return;
    }

    index_t jy = start_index(n, incy);
    for (int j = 0; j < n; ++j) {
        T temp = alpha * y[jy];
        index_t ix = start_index(m, incx);
        for (int i = 0; i < m; ++i) {
            a[idx(i, j, lda)] += x[ix] * temp;
            ix += incx;
        }
        jy += incy;
    }
}

/* ------------------------------------------------------------------ */
/* geru_impl — a ← α·x·yᵀ + a   (complex unconjugated rank-1)      */
/* ------------------------------------------------------------------ */
template <typename T>
void geru_impl(const char *routine, int m, int n, std::complex<T> alpha, const std::complex<T> *x,
               int incx, const std::complex<T> *y, int incy, std::complex<T> *a, int lda) {
    using C = std::complex<T>;
    const int info = ger_info(m, n, incx, incy, lda);
    if (info != 0) {
        report_error(routine, info);
        return;
    }
    if (m == 0 || n == 0) {
        return;
    }
    if (alpha == C(0)) {
        return;
    }

    index_t jy = start_index(n, incy);
    for (int j = 0; j < n; ++j) {
        C temp = alpha * y[jy];
        index_t ix = start_index(m, incx);
        for (int i = 0; i < m; ++i) {
            a[idx(i, j, lda)] += x[ix] * temp;
            ix += incx;
        }
        jy += incy;
    }
}

/* ------------------------------------------------------------------ */
/* gerc_impl — a ← α·x·conj(y)ᵀ + a   (complex conjugated rank-1)  */
/* ------------------------------------------------------------------ */
template <typename T>
void gerc_impl(const char *routine, int m, int n, std::complex<T> alpha, const std::complex<T> *x,
               int incx, const std::complex<T> *y, int incy, std::complex<T> *a, int lda) {
    using C = std::complex<T>;
    const int info = ger_info(m, n, incx, incy, lda);
    if (info != 0) {
        report_error(routine, info);
        return;
    }
    if (m == 0 || n == 0) {
        return;
    }
    if (alpha == C(0)) {
        return;
    }

    index_t jy = start_index(n, incy);
    for (int j = 0; j < n; ++j) {
        C temp = alpha * std::conj(y[jy]);
        index_t ix = start_index(m, incx);
        for (int i = 0; i < m; ++i) {
            a[idx(i, j, lda)] += x[ix] * temp;
            ix += incx;
        }
        jy += incy;
    }
}

/* ------------------------------------------------------------------ */
/* syr_impl — a ← α·x·xᵀ + a   (a symmetric)                       */
/* ------------------------------------------------------------------ */
template <typename T>
void syr_impl(const char *routine, char uplo, int n, T alpha, const T *x, int incx, T *a, int lda) {
    const int info = syr_info(uplo, n, incx, lda);
    if (info != 0) {
        report_error(routine, info);
        return;
    }
    if (n == 0) {
        return;
    }
    char ul = to_upper(uplo);
    if (alpha == T(0)) {
        return;
    }

    index_t jx = start_index(n, incx);
    if (ul == 'U') {
        for (int j = 0; j < n; ++j) {
            T temp = alpha * x[jx];
            index_t ix = start_index(n, incx);
            for (int i = 0; i <= j; ++i) {
                a[idx(i, j, lda)] += x[ix] * temp;
                ix += incx;
            }
            jx += incx;
        }
    } else {
        for (int j = 0; j < n; ++j) {
            T temp = alpha * x[jx];
            index_t ix = jx;
            for (int i = j; i < n; ++i) {
                a[idx(i, j, lda)] += x[ix] * temp;
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
void her_impl(const char *routine, char uplo, int n, T alpha, const std::complex<T> *x, int incx,
              std::complex<T> *a, int lda) {
    using C = std::complex<T>;
    const int info = syr_info(uplo, n, incx, lda);
    if (info != 0) {
        report_error(routine, info);
        return;
    }
    if (n == 0) {
        return;
    }
    char ul = to_upper(uplo);
    if (alpha == T(0)) {
        return;
    }

    index_t jx = start_index(n, incx);
    if (ul == 'U') {
        for (int j = 0; j < n; ++j) {
            C temp = C(alpha) * std::conj(x[jx]);
            index_t ix = start_index(n, incx);
            for (int i = 0; i < j; ++i) {
                a[idx(i, j, lda)] += x[ix] * temp;
                ix += incx;
            }
            // Diagonal is real
            a[idx(j, j, lda)] = C(a[idx(j, j, lda)].real() + (x[jx] * temp).real());
            jx += incx;
        }
    } else {
        for (int j = 0; j < n; ++j) {
            C temp = C(alpha) * std::conj(x[jx]);
            // Diagonal is real
            a[idx(j, j, lda)] = C(a[idx(j, j, lda)].real() + (x[jx] * temp).real());
            index_t ix = jx;
            for (int i = j + 1; i < n; ++i) {
                ix += incx;
                a[idx(i, j, lda)] += x[ix] * temp;
            }
            jx += incx;
        }
    }
}

/* ------------------------------------------------------------------ */
/* syr2_impl — a ← α·x·yᵀ + α·y·xᵀ + a   (a symmetric)            */
/* ------------------------------------------------------------------ */
template <typename T>
void syr2_impl(const char *routine, char uplo, int n, T alpha, const T *x, int incx, const T *y,
               int incy, T *a, int lda) {
    const int info = syr2_info(uplo, n, incx, incy, lda);
    if (info != 0) {
        report_error(routine, info);
        return;
    }
    if (n == 0) {
        return;
    }
    char ul = to_upper(uplo);
    if (alpha == T(0)) {
        return;
    }

    index_t jx = start_index(n, incx);
    index_t jy = start_index(n, incy);
    if (ul == 'U') {
        for (int j = 0; j < n; ++j) {
            T temp1 = alpha * y[jy];
            T temp2 = alpha * x[jx];
            index_t ix = start_index(n, incx);
            index_t iy = start_index(n, incy);
            for (int i = 0; i <= j; ++i) {
                a[idx(i, j, lda)] += (x[ix] * temp1) + (y[iy] * temp2);
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
            index_t ix = jx;
            index_t iy = jy;
            for (int i = j; i < n; ++i) {
                a[idx(i, j, lda)] += (x[ix] * temp1) + (y[iy] * temp2);
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
void her2_impl(const char *routine, char uplo, int n, std::complex<T> alpha,
               const std::complex<T> *x, int incx, const std::complex<T> *y, int incy,
               std::complex<T> *a, int lda) {
    using C = std::complex<T>;
    const int info = syr2_info(uplo, n, incx, incy, lda);
    if (info != 0) {
        report_error(routine, info);
        return;
    }
    if (n == 0) {
        return;
    }
    char ul = to_upper(uplo);
    if (alpha == C(0)) {
        return;
    }

    index_t jx = start_index(n, incx);
    index_t jy = start_index(n, incy);
    if (ul == 'U') {
        for (int j = 0; j < n; ++j) {
            C temp1 = alpha * std::conj(y[jy]);
            C temp2 = std::conj(alpha * x[jx]);
            index_t ix = start_index(n, incx);
            index_t iy = start_index(n, incy);
            for (int i = 0; i < j; ++i) {
                a[idx(i, j, lda)] += (x[ix] * temp1) + (y[iy] * temp2);
                ix += incx;
                iy += incy;
            }
            a[idx(j, j, lda)] = C((a[idx(j, j, lda)] + (x[jx] * temp1) + (y[jy] * temp2)).real());
            jx += incx;
            jy += incy;
        }
    } else {
        for (int j = 0; j < n; ++j) {
            C temp1 = alpha * std::conj(y[jy]);
            C temp2 = std::conj(alpha * x[jx]);
            a[idx(j, j, lda)] = C((a[idx(j, j, lda)] + (x[jx] * temp1) + (y[jy] * temp2)).real());
            index_t ix = jx;
            index_t iy = jy;
            for (int i = j + 1; i < n; ++i) {
                ix += incx;
                iy += incy;
                a[idx(i, j, lda)] += (x[ix] * temp1) + (y[iy] * temp2);
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

void sgemv(char trans, int m, int n, float alpha, const float *a, int lda, const float *x, int incx,
           float beta, float *y, int incy) {
    gemv_impl("SGEMV", trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
void dgemv(char trans, int m, int n, double alpha, const double *a, int lda, const double *x,
           int incx, double beta, double *y, int incy) {
    gemv_impl("DGEMV", trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
void cgemv(char trans, int m, int n, std::complex<float> alpha, const std::complex<float> *a,
           int lda, const std::complex<float> *x, int incx, std::complex<float> beta,
           std::complex<float> *y, int incy) {
    gemv_impl("CGEMV", trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
void zgemv(char trans, int m, int n, std::complex<double> alpha, const std::complex<double> *a,
           int lda, const std::complex<double> *x, int incx, std::complex<double> beta,
           std::complex<double> *y, int incy) {
    gemv_impl("ZGEMV", trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

/* symv */
void ssymv(char uplo, int n, float alpha, const float *a, int lda, const float *x, int incx,
           float beta, float *y, int incy) {
    symv_impl("SSYMV", uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}
void dsymv(char uplo, int n, double alpha, const double *a, int lda, const double *x, int incx,
           double beta, double *y, int incy) {
    symv_impl("DSYMV", uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

/* hemv */
void chemv(char uplo, int n, std::complex<float> alpha, const std::complex<float> *a, int lda,
           const std::complex<float> *x, int incx, std::complex<float> beta, std::complex<float> *y,
           int incy) {
    hemv_impl("CHEMV", uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}
void zhemv(char uplo, int n, std::complex<double> alpha, const std::complex<double> *a, int lda,
           const std::complex<double> *x, int incx, std::complex<double> beta,
           std::complex<double> *y, int incy) {
    hemv_impl("ZHEMV", uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

/* trmv */
void strmv(char uplo, char trans, char diag, int n, const float *a, int lda, float *x, int incx) {
    trmv_impl("STRMV", uplo, trans, diag, n, a, lda, x, incx);
}
void dtrmv(char uplo, char trans, char diag, int n, const double *a, int lda, double *x, int incx) {
    trmv_impl("DTRMV", uplo, trans, diag, n, a, lda, x, incx);
}
void ctrmv(char uplo, char trans, char diag, int n, const std::complex<float> *a, int lda,
           std::complex<float> *x, int incx) {
    trmv_impl("CTRMV", uplo, trans, diag, n, a, lda, x, incx);
}
void ztrmv(char uplo, char trans, char diag, int n, const std::complex<double> *a, int lda,
           std::complex<double> *x, int incx) {
    trmv_impl("ZTRMV", uplo, trans, diag, n, a, lda, x, incx);
}

/* trsv */
void strsv(char uplo, char trans, char diag, int n, const float *a, int lda, float *x, int incx) {
    trsv_impl("STRSV", uplo, trans, diag, n, a, lda, x, incx);
}
void dtrsv(char uplo, char trans, char diag, int n, const double *a, int lda, double *x, int incx) {
    trsv_impl("DTRSV", uplo, trans, diag, n, a, lda, x, incx);
}
void ctrsv(char uplo, char trans, char diag, int n, const std::complex<float> *a, int lda,
           std::complex<float> *x, int incx) {
    trsv_impl("CTRSV", uplo, trans, diag, n, a, lda, x, incx);
}
void ztrsv(char uplo, char trans, char diag, int n, const std::complex<double> *a, int lda,
           std::complex<double> *x, int incx) {
    trsv_impl("ZTRSV", uplo, trans, diag, n, a, lda, x, incx);
}

/* ger */
void sger(int m, int n, float alpha, const float *x, int incx, const float *y, int incy, float *a,
          int lda) {
    ger_impl("SGER", m, n, alpha, x, incx, y, incy, a, lda);
}
void dger(int m, int n, double alpha, const double *x, int incx, const double *y, int incy,
          double *a, int lda) {
    ger_impl("DGER", m, n, alpha, x, incx, y, incy, a, lda);
}

/* geru / gerc */
void cgeru(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int incx,
           const std::complex<float> *y, int incy, std::complex<float> *a, int lda) {
    geru_impl("CGERU", m, n, alpha, x, incx, y, incy, a, lda);
}
void zgeru(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int incx,
           const std::complex<double> *y, int incy, std::complex<double> *a, int lda) {
    geru_impl("ZGERU", m, n, alpha, x, incx, y, incy, a, lda);
}
void cgerc(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int incx,
           const std::complex<float> *y, int incy, std::complex<float> *a, int lda) {
    gerc_impl("CGERC", m, n, alpha, x, incx, y, incy, a, lda);
}
void zgerc(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int incx,
           const std::complex<double> *y, int incy, std::complex<double> *a, int lda) {
    gerc_impl("ZGERC", m, n, alpha, x, incx, y, incy, a, lda);
}

/* syr */
void ssyr(char uplo, int n, float alpha, const float *x, int incx, float *a, int lda) {
    syr_impl("SSYR", uplo, n, alpha, x, incx, a, lda);
}
void dsyr(char uplo, int n, double alpha, const double *x, int incx, double *a, int lda) {
    syr_impl("DSYR", uplo, n, alpha, x, incx, a, lda);
}

/* her */
void cher(char uplo, int n, float alpha, const std::complex<float> *x, int incx,
          std::complex<float> *a, int lda) {
    her_impl("CHER", uplo, n, alpha, x, incx, a, lda);
}
void zher(char uplo, int n, double alpha, const std::complex<double> *x, int incx,
          std::complex<double> *a, int lda) {
    her_impl("ZHER", uplo, n, alpha, x, incx, a, lda);
}

/* syr2 */
void ssyr2(char uplo, int n, float alpha, const float *x, int incx, const float *y, int incy,
           float *a, int lda) {
    syr2_impl("SSYR2", uplo, n, alpha, x, incx, y, incy, a, lda);
}
void dsyr2(char uplo, int n, double alpha, const double *x, int incx, const double *y, int incy,
           double *a, int lda) {
    syr2_impl("DSYR2", uplo, n, alpha, x, incx, y, incy, a, lda);
}

/* her2 */
void cher2(char uplo, int n, std::complex<float> alpha, const std::complex<float> *x, int incx,
           const std::complex<float> *y, int incy, std::complex<float> *a, int lda) {
    her2_impl("CHER2", uplo, n, alpha, x, incx, y, incy, a, lda);
}
void zher2(char uplo, int n, std::complex<double> alpha, const std::complex<double> *x, int incx,
           const std::complex<double> *y, int incy, std::complex<double> *a, int lda) {
    her2_impl("ZHER2", uplo, n, alpha, x, incx, y, incy, a, lda);
}

/* gbmv */

} // namespace theblas
