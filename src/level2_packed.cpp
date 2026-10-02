#include "theblas/theblas.h"

#include "theblas_detail.hpp"

#include <algorithm>
#include <complex>

namespace theblas {

using detail::conj_val;
using detail::index_t;
using detail::offset;
using detail::packed_size;
using detail::report_error;
using detail::start_index;
using detail::to_upper;
using detail::valid_diag;
using detail::valid_trans;
using detail::valid_uplo;

namespace {

int spmv_info(char uplo, int n, int incx, int incy) {
    if (!valid_uplo(uplo)) {
        return 1;
    }
    if (n < 0) {
        return 2;
    }
    if (incx == 0) {
        return 6;
    }
    if (incy == 0) {
        return 9;
    }
    return 0;
}

int tpmv_info(char uplo, char trans, char diag, int n, int incx) {
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
    if (incx == 0) {
        return 7;
    }
    return 0;
}

int spr_info(char uplo, int n, int incx) {
    if (!valid_uplo(uplo)) {
        return 1;
    }
    if (n < 0) {
        return 2;
    }
    if (incx == 0) {
        return 5;
    }
    return 0;
}

int spr2_info(char uplo, int n, int incx, int incy) {
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
    return 0;
}

template <typename T>
void spmv_impl(const char *routine, char uplo, int n, T alpha, const T *ap, const T *x, int incx,
               T beta, T *y, int incy) {
    const int info = spmv_info(uplo, n, incx, incy);
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

    index_t kk = 0;
    if (ul == 'U') {
        index_t jx = start_index(n, incx);
        index_t jy = start_index(n, incy);
        for (int j = 0; j < n; ++j) {
            T temp1 = alpha * x[jx];
            T temp2 = T(0);
            index_t ix = start_index(n, incx);
            index_t iy = start_index(n, incy);
            for (int i = 0; i < j; ++i) {
                y[iy] += temp1 * ap[kk + i];
                temp2 += ap[kk + i] * x[ix];
                ix += incx;
                iy += incy;
            }
            y[jy] += (temp1 * ap[kk + j]) + (alpha * temp2);
            kk += j + 1;
            jx += incx;
            jy += incy;
        }
    } else {
        index_t jx = start_index(n, incx);
        index_t jy = start_index(n, incy);
        for (int j = 0; j < n; ++j) {
            T temp1 = alpha * x[jx];
            T temp2 = T(0);
            y[jy] += temp1 * ap[kk];
            index_t ix = jx;
            index_t iy = jy;
            for (int i = j + 1; i < n; ++i) {
                ix += incx;
                iy += incy;
                y[iy] += temp1 * ap[kk + i - j];
                temp2 += ap[kk + i - j] * x[ix];
            }
            y[jy] += alpha * temp2;
            kk += n - j;
            jx += incx;
            jy += incy;
        }
    }
}

/* ------------------------------------------------------------------ */
/* hpmv_impl — y ← α·a·x + β·y   (a Hermitian, packed)             */
/* ------------------------------------------------------------------ */
template <typename T>
void hpmv_impl(const char *routine, char uplo, int n, std::complex<T> alpha,
               const std::complex<T> *ap, const std::complex<T> *x, int incx, std::complex<T> beta,
               std::complex<T> *y, int incy) {
    using C = std::complex<T>;
    const int info = spmv_info(uplo, n, incx, incy);
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

    index_t kk = 0;
    if (ul == 'U') {
        index_t jx = start_index(n, incx);
        index_t jy = start_index(n, incy);
        for (int j = 0; j < n; ++j) {
            C temp1 = alpha * x[jx];
            C temp2 = C(0);
            index_t ix = start_index(n, incx);
            index_t iy = start_index(n, incy);
            for (int i = 0; i < j; ++i) {
                y[iy] += temp1 * ap[kk + i];
                temp2 += std::conj(ap[kk + i]) * x[ix];
                ix += incx;
                iy += incy;
            }
            y[jy] += (temp1 * T(ap[kk + j].real())) + (alpha * temp2);
            kk += j + 1;
            jx += incx;
            jy += incy;
        }
    } else {
        index_t jx = start_index(n, incx);
        index_t jy = start_index(n, incy);
        for (int j = 0; j < n; ++j) {
            C temp1 = alpha * x[jx];
            C temp2 = C(0);
            y[jy] += temp1 * T(ap[kk].real());
            index_t ix = jx;
            index_t iy = jy;
            for (int i = j + 1; i < n; ++i) {
                ix += incx;
                iy += incy;
                y[iy] += temp1 * ap[kk + i - j];
                temp2 += std::conj(ap[kk + i - j]) * x[ix];
            }
            y[jy] += alpha * temp2;
            kk += n - j;
            jx += incx;
            jy += incy;
        }
    }
}

/* ------------------------------------------------------------------ */
/* tpmv_impl — x ← op(a)·x   (a triangular, packed)                 */
/* ------------------------------------------------------------------ */
template <typename T>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void tpmv_impl(const char *routine, char uplo, char trans, char diag, int n, const T *ap, T *x,
               int incx) {
    const int info = tpmv_info(uplo, trans, diag, n, incx);
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
            index_t kk = 0;
            index_t jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                if (x[jx] != T(0)) {
                    T temp = x[jx];
                    index_t ix = start_index(n, incx);
                    for (int i = 0; i < j; ++i) {
                        x[ix] += temp * ap[kk + i];
                        ix += incx;
                    }
                    if (!unit) {
                        x[jx] *= ap[kk + j];
                    }
                }
                kk += j + 1;
                jx += incx;
            }
        } else {
            index_t kk = packed_size(n) - 1;
            index_t jx = start_index(n, incx) + offset(n - 1, incx);
            for (int j = n - 1; j >= 0; --j) {
                if (x[jx] != T(0)) {
                    T temp = x[jx];
                    index_t ix = start_index(n, incx) + offset(n - 1, incx);
                    for (int i = n - 1; i > j; --i) {
                        x[ix] += temp * ap[kk - (n - 1 - i)];
                        ix -= incx;
                    }
                    if (!unit) {
                        x[jx] *= ap[kk - (n - 1 - j)];
                    }
                }
                kk -= n - j;
                jx -= incx;
            }
        }
    } else {
        // Transpose or conjugate-transpose
        if (ul == 'U') {
            index_t kk = packed_size(n) - 1;
            index_t jx = start_index(n, incx) + offset(n - 1, incx);
            for (int j = n - 1; j >= 0; --j) {
                T temp = x[jx];
                if (!unit) {
                    T aval = conj ? conj_val(ap[kk]) : ap[kk];
                    temp *= aval;
                }
                index_t ix = jx;
                index_t k = kk - 1;
                for (int i = j - 1; i >= 0; --i) {
                    ix -= incx;
                    T aval = conj ? conj_val(ap[k]) : ap[k];
                    temp += aval * x[ix];
                    --k;
                }
                x[jx] = temp;
                kk -= j + 1;
                jx -= incx;
            }
        } else {
            index_t kk = 0;
            index_t jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                T temp = x[jx];
                if (!unit) {
                    T aval = conj ? conj_val(ap[kk]) : ap[kk];
                    temp *= aval;
                }
                index_t ix = jx;
                index_t k = kk + 1;
                for (int i = j + 1; i < n; ++i) {
                    ix += incx;
                    T aval = conj ? conj_val(ap[k]) : ap[k];
                    temp += aval * x[ix];
                    ++k;
                }
                x[jx] = temp;
                kk += n - j;
                jx += incx;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* tpsv_impl — solve op(a)·x = b   (a triangular, packed)            */
/* ------------------------------------------------------------------ */
template <typename T>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void tpsv_impl(const char *routine, char uplo, char trans, char diag, int n, const T *ap, T *x,
               int incx) {
    const int info = tpmv_info(uplo, trans, diag, n, incx);
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
            index_t kk = packed_size(n) - 1;
            index_t jx = start_index(n, incx) + offset(n - 1, incx);
            for (int j = n - 1; j >= 0; --j) {
                if (!unit) {
                    x[jx] /= ap[kk];
                }
                T temp = x[jx];
                index_t ix = jx;
                index_t k = kk - 1;
                for (int i = j - 1; i >= 0; --i) {
                    ix -= incx;
                    x[ix] -= temp * ap[k];
                    --k;
                }
                kk -= j + 1;
                jx -= incx;
            }
        } else {
            index_t kk = 0;
            index_t jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                if (!unit) {
                    x[jx] /= ap[kk];
                }
                T temp = x[jx];
                index_t ix = jx;
                index_t k = kk + 1;
                for (int i = j + 1; i < n; ++i) {
                    ix += incx;
                    x[ix] -= temp * ap[k];
                    ++k;
                }
                kk += n - j;
                jx += incx;
            }
        }
    } else {
        if (ul == 'U') {
            index_t kk = 0;
            index_t jx = start_index(n, incx);
            for (int j = 0; j < n; ++j) {
                T temp = x[jx];
                index_t ix = start_index(n, incx);
                index_t k = kk;
                for (int i = 0; i < j; ++i) {
                    T aval = conj ? conj_val(ap[k]) : ap[k];
                    temp -= aval * x[ix];
                    ix += incx;
                    ++k;
                }
                if (!unit) {
                    T aval = conj ? conj_val(ap[k]) : ap[k];
                    temp /= aval;
                }
                x[jx] = temp;
                kk += j + 1;
                jx += incx;
            }
        } else {
            index_t kk = packed_size(n) - 1;
            index_t jx = start_index(n, incx) + offset(n - 1, incx);
            for (int j = n - 1; j >= 0; --j) {
                T temp = x[jx];
                index_t ix = start_index(n, incx) + offset(n - 1, incx);
                index_t k = kk;
                for (int i = n - 1; i > j; --i) {
                    T aval = conj ? conj_val(ap[k]) : ap[k];
                    temp -= aval * x[ix];
                    ix -= incx;
                    --k;
                }
                if (!unit) {
                    T aval = conj ? conj_val(ap[k]) : ap[k];
                    temp /= aval;
                }
                x[jx] = temp;
                kk -= n - j;
                jx -= incx;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* spr_impl — a ← α·x·xᵀ + a   (a symmetric, packed)               */
/* ------------------------------------------------------------------ */
template <typename T>
void spr_impl(const char *routine, char uplo, int n, T alpha, const T *x, int incx, T *ap) {
    const int info = spr_info(uplo, n, incx);
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

    index_t kk = 0;
    index_t jx = start_index(n, incx);
    if (ul == 'U') {
        for (int j = 0; j < n; ++j) {
            T temp = alpha * x[jx];
            index_t ix = start_index(n, incx);
            for (int i = 0; i <= j; ++i) {
                ap[kk + i] += x[ix] * temp;
                ix += incx;
            }
            kk += j + 1;
            jx += incx;
        }
    } else {
        for (int j = 0; j < n; ++j) {
            T temp = alpha * x[jx];
            index_t ix = jx;
            for (int i = j; i < n; ++i) {
                ap[kk + i - j] += x[ix] * temp;
                ix += incx;
            }
            kk += n - j;
            jx += incx;
        }
    }
}

/* ------------------------------------------------------------------ */
/* hpr_impl — a ← α·x·xᴴ + a   (a Hermitian, packed, α real)       */
/* ------------------------------------------------------------------ */
template <typename T>
void hpr_impl(const char *routine, char uplo, int n, T alpha, const std::complex<T> *x, int incx,
              std::complex<T> *ap) {
    using C = std::complex<T>;
    const int info = spr_info(uplo, n, incx);
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

    index_t kk = 0;
    index_t jx = start_index(n, incx);
    if (ul == 'U') {
        for (int j = 0; j < n; ++j) {
            C temp = C(alpha) * std::conj(x[jx]);
            index_t ix = start_index(n, incx);
            for (int i = 0; i < j; ++i) {
                ap[kk + i] += x[ix] * temp;
                ix += incx;
            }
            ap[kk + j] = C(ap[kk + j].real() + (x[jx] * temp).real());
            kk += j + 1;
            jx += incx;
        }
    } else {
        for (int j = 0; j < n; ++j) {
            C temp = C(alpha) * std::conj(x[jx]);
            ap[kk] = C(ap[kk].real() + (x[jx] * temp).real());
            index_t ix = jx;
            for (int i = j + 1; i < n; ++i) {
                ix += incx;
                ap[kk + i - j] += x[ix] * temp;
            }
            kk += n - j;
            jx += incx;
        }
    }
}

/* ------------------------------------------------------------------ */
/* spr2_impl — a ← α·x·yᵀ + α·y·xᵀ + a   (symmetric packed)       */
/* ------------------------------------------------------------------ */
template <typename T>
void spr2_impl(const char *routine, char uplo, int n, T alpha, const T *x, int incx, const T *y,
               int incy, T *ap) {
    const int info = spr2_info(uplo, n, incx, incy);
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

    index_t kk = 0;
    index_t jx = start_index(n, incx);
    index_t jy = start_index(n, incy);
    if (ul == 'U') {
        for (int j = 0; j < n; ++j) {
            T temp1 = alpha * y[jy];
            T temp2 = alpha * x[jx];
            index_t ix = start_index(n, incx);
            index_t iy = start_index(n, incy);
            for (int i = 0; i <= j; ++i) {
                ap[kk + i] += (x[ix] * temp1) + (y[iy] * temp2);
                ix += incx;
                iy += incy;
            }
            kk += j + 1;
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
                ap[kk + i - j] += (x[ix] * temp1) + (y[iy] * temp2);
                ix += incx;
                iy += incy;
            }
            kk += n - j;
            jx += incx;
            jy += incy;
        }
    }
}

/* ------------------------------------------------------------------ */
/* hpr2_impl — a ← α·x·yᴴ + conj(α)·y·xᴴ + a   (Hermitian packed) */
/* ------------------------------------------------------------------ */
template <typename T>
void hpr2_impl(const char *routine, char uplo, int n, std::complex<T> alpha,
               const std::complex<T> *x, int incx, const std::complex<T> *y, int incy,
               std::complex<T> *ap) {
    using C = std::complex<T>;
    const int info = spr2_info(uplo, n, incx, incy);
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

    index_t kk = 0;
    index_t jx = start_index(n, incx);
    index_t jy = start_index(n, incy);
    if (ul == 'U') {
        for (int j = 0; j < n; ++j) {
            C temp1 = alpha * std::conj(y[jy]);
            C temp2 = std::conj(alpha * x[jx]);
            index_t ix = start_index(n, incx);
            index_t iy = start_index(n, incy);
            for (int i = 0; i < j; ++i) {
                ap[kk + i] += (x[ix] * temp1) + (y[iy] * temp2);
                ix += incx;
                iy += incy;
            }
            ap[kk + j] = C((ap[kk + j] + (x[jx] * temp1) + (y[jy] * temp2)).real());
            kk += j + 1;
            jx += incx;
            jy += incy;
        }
    } else {
        for (int j = 0; j < n; ++j) {
            C temp1 = alpha * std::conj(y[jy]);
            C temp2 = std::conj(alpha * x[jx]);
            ap[kk] = C((ap[kk] + (x[jx] * temp1) + (y[jy] * temp2)).real());
            index_t ix = jx;
            index_t iy = jy;
            for (int i = j + 1; i < n; ++i) {
                ix += incx;
                iy += incy;
                ap[kk + i - j] += (x[ix] * temp1) + (y[iy] * temp2);
            }
            kk += n - j;
            jx += incx;
            jy += incy;
        }
    }
}

} // anonymous namespace

void sspmv(char uplo, int n, float alpha, const float *ap, const float *x, int incx, float beta,
           float *y, int incy) {
    spmv_impl("SSPMV", uplo, n, alpha, ap, x, incx, beta, y, incy);
}
void dspmv(char uplo, int n, double alpha, const double *ap, const double *x, int incx, double beta,
           double *y, int incy) {
    spmv_impl("DSPMV", uplo, n, alpha, ap, x, incx, beta, y, incy);
}

/* hpmv */
void chpmv(char uplo, int n, std::complex<float> alpha, const std::complex<float> *ap,
           const std::complex<float> *x, int incx, std::complex<float> beta, std::complex<float> *y,
           int incy) {
    hpmv_impl("CHPMV", uplo, n, alpha, ap, x, incx, beta, y, incy);
}
void zhpmv(char uplo, int n, std::complex<double> alpha, const std::complex<double> *ap,
           const std::complex<double> *x, int incx, std::complex<double> beta,
           std::complex<double> *y, int incy) {
    hpmv_impl("ZHPMV", uplo, n, alpha, ap, x, incx, beta, y, incy);
}

/* tpmv */
void stpmv(char uplo, char trans, char diag, int n, const float *ap, float *x, int incx) {
    tpmv_impl("STPMV", uplo, trans, diag, n, ap, x, incx);
}
void dtpmv(char uplo, char trans, char diag, int n, const double *ap, double *x, int incx) {
    tpmv_impl("DTPMV", uplo, trans, diag, n, ap, x, incx);
}
void ctpmv(char uplo, char trans, char diag, int n, const std::complex<float> *ap,
           std::complex<float> *x, int incx) {
    tpmv_impl("CTPMV", uplo, trans, diag, n, ap, x, incx);
}
void ztpmv(char uplo, char trans, char diag, int n, const std::complex<double> *ap,
           std::complex<double> *x, int incx) {
    tpmv_impl("ZTPMV", uplo, trans, diag, n, ap, x, incx);
}

/* tpsv */
void stpsv(char uplo, char trans, char diag, int n, const float *ap, float *x, int incx) {
    tpsv_impl("STPSV", uplo, trans, diag, n, ap, x, incx);
}
void dtpsv(char uplo, char trans, char diag, int n, const double *ap, double *x, int incx) {
    tpsv_impl("DTPSV", uplo, trans, diag, n, ap, x, incx);
}
void ctpsv(char uplo, char trans, char diag, int n, const std::complex<float> *ap,
           std::complex<float> *x, int incx) {
    tpsv_impl("CTPSV", uplo, trans, diag, n, ap, x, incx);
}
void ztpsv(char uplo, char trans, char diag, int n, const std::complex<double> *ap,
           std::complex<double> *x, int incx) {
    tpsv_impl("ZTPSV", uplo, trans, diag, n, ap, x, incx);
}

/* spr */
void sspr(char uplo, int n, float alpha, const float *x, int incx, float *ap) {
    spr_impl("SSPR", uplo, n, alpha, x, incx, ap);
}
void dspr(char uplo, int n, double alpha, const double *x, int incx, double *ap) {
    spr_impl("DSPR", uplo, n, alpha, x, incx, ap);
}

/* hpr */
void chpr(char uplo, int n, float alpha, const std::complex<float> *x, int incx,
          std::complex<float> *ap) {
    hpr_impl("CHPR", uplo, n, alpha, x, incx, ap);
}
void zhpr(char uplo, int n, double alpha, const std::complex<double> *x, int incx,
          std::complex<double> *ap) {
    hpr_impl("ZHPR", uplo, n, alpha, x, incx, ap);
}

/* spr2 */
void sspr2(char uplo, int n, float alpha, const float *x, int incx, const float *y, int incy,
           float *ap) {
    spr2_impl("SSPR2", uplo, n, alpha, x, incx, y, incy, ap);
}
void dspr2(char uplo, int n, double alpha, const double *x, int incx, const double *y, int incy,
           double *ap) {
    spr2_impl("DSPR2", uplo, n, alpha, x, incx, y, incy, ap);
}

/* hpr2 */
void chpr2(char uplo, int n, std::complex<float> alpha, const std::complex<float> *x, int incx,
           const std::complex<float> *y, int incy, std::complex<float> *ap) {
    hpr2_impl("CHPR2", uplo, n, alpha, x, incx, y, incy, ap);
}
void zhpr2(char uplo, int n, std::complex<double> alpha, const std::complex<double> *x, int incx,
           const std::complex<double> *y, int incy, std::complex<double> *ap) {
    hpr2_impl("ZHPR2", uplo, n, alpha, x, incx, y, incy, ap);
}

} // namespace theblas
