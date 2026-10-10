#include "theblas/theblas.h"

#include "theblas_detail.hpp"

#include <algorithm>
#include <complex>
#include <vector>

// BLAS kernels preserve the standard ordering of flags, dimensions, and strides.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)

namespace theblas {

using detail::conj_val;
using detail::idx;
using detail::index_t;
using detail::report_error;
using detail::to_upper;
using detail::valid_diag;
using detail::valid_side;
using detail::valid_trans;
using detail::valid_uplo;

namespace {

template <typename T> T op_value(const T *a, int lda, char trans, int row, int col) {
    const char tr = to_upper(trans);
    if (tr == 'N') {
        return a[idx(row, col, lda)];
    }
    const T value = a[idx(col, row, lda)];
    return tr == 'C' ? conj_val(value) : value;
}

bool valid_rank_trans(char trans, bool hermitian) {
    const char upper = to_upper(trans);
    return upper == 'N' || (hermitian ? upper == 'C' : upper == 'T');
}

template <typename T> std::complex<T> real_diagonal(std::complex<T> value) {
    return {value.real(), T(0)};
}

template <typename T, bool Hermitian>
T symmetric_value(const T *a, int lda, char uplo, int row, int col) {
    if (row == col) {
        if constexpr (Hermitian) {
            return real_diagonal(a[idx(row, col, lda)]);
        } else {
            return a[idx(row, col, lda)];
        }
    }
    if (to_upper(uplo) == 'U' ? row < col : row > col) {
        return a[idx(row, col, lda)];
    }
    const T value = a[idx(col, row, lda)];
    if constexpr (Hermitian) {
        return conj_val(value);
    } else {
        return value;
    }
}

template <typename T> void scale_matrix(int m, int n, T beta, T *c, int ldc) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            auto &value = c[idx(i, j, ldc)];
            if (beta == T(0)) {
                value = T(0);
            } else if (beta != T(1)) {
                value *= beta;
            }
        }
    }
}

int gemm_info(char ta, char tb, int m, int n, int k, int lda, int ldb, int ldc) {
    if (!valid_trans(ta)) {
        return 1;
    }
    if (!valid_trans(tb)) {
        return 2;
    }
    if (m < 0) {
        return 3;
    }
    if (n < 0) {
        return 4;
    }
    if (k < 0) {
        return 5;
    }
    if (lda < std::max(1, to_upper(ta) == 'N' ? m : k)) {
        return 8;
    }
    if (ldb < std::max(1, to_upper(tb) == 'N' ? k : n)) {
        return 10;
    }
    if (ldc < std::max(1, m)) {
        return 13;
    }
    return 0;
}

template <typename T>
void gemm_impl(const char *routine, char transa, char transb, int m, int n, int k, T alpha,
               const T *a, int lda, const T *b, int ldb, T beta, T *c, int ldc) {
    const int info = gemm_info(transa, transb, m, n, k, lda, ldb, ldc);
    if (info != 0) {
        report_error(routine, info);
        return;
    }
    if (m == 0 || n == 0) {
        return;
    }
    if (alpha == T(0) || k == 0) {
        if (beta == T(1)) {
            return;
        }
        scale_matrix(m, n, beta, c, ldc);
        return;
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            T sum = T(0);
            for (int l = 0; l < k; ++l) {
                sum += op_value(a, lda, transa, i, l) * op_value(b, ldb, transb, l, j);
            }
            auto &value = c[idx(i, j, ldc)];
            value = alpha * sum + (beta == T(0) ? T(0) : beta * value);
        }
    }
}

int side_matrix_info(char side, char uplo, int m, int n, int lda, int ldb, int ldc) {
    if (!valid_side(side)) {
        return 1;
    }
    if (!valid_uplo(uplo)) {
        return 2;
    }
    if (m < 0) {
        return 3;
    }
    if (n < 0) {
        return 4;
    }
    const int order = to_upper(side) == 'L' ? m : n;
    if (lda < std::max(1, order)) {
        return 7;
    }
    if (ldb < std::max(1, m)) {
        return 9;
    }
    if (ldc < std::max(1, m)) {
        return 12;
    }
    return 0;
}

template <typename T, bool Hermitian>
void symm_impl(const char *routine, char side, char uplo, int m, int n, T alpha, const T *a,
               int lda, const T *b, int ldb, T beta, T *c, int ldc) {
    const int info = side_matrix_info(side, uplo, m, n, lda, ldb, ldc);
    if (info != 0) {
        report_error(routine, info);
        return;
    }
    if (m == 0 || n == 0) {
        return;
    }
    if (alpha == T(0)) {
        if (beta == T(1)) {
            return;
        }
        scale_matrix(m, n, beta, c, ldc);
        return;
    }
    const bool left = to_upper(side) == 'L';
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            T sum = T(0);
            const int order = left ? m : n;
            for (int l = 0; l < order; ++l) {
                if (left) {
                    sum += symmetric_value<T, Hermitian>(a, lda, uplo, i, l) * b[idx(l, j, ldb)];
                } else {
                    sum += b[idx(i, l, ldb)] *
                          symmetric_value<T, Hermitian>(a, lda, uplo, l, j);
                }
            }
            auto &value = c[idx(i, j, ldc)];
            value = alpha * sum + (beta == T(0) ? T(0) : beta * value);
        }
    }
}

template <typename T, typename Scalar, bool Hermitian>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void syrk_impl(const char *routine, char uplo, char trans, int n, int k, Scalar alpha, const T *a,
               int lda, Scalar beta, T *c, int ldc) {
    if (!valid_uplo(uplo)) {
        report_error(routine, 1);
        return;
    }
    if (!valid_rank_trans(trans, Hermitian)) {
        report_error(routine, 2);
        return;
    }
    if (n < 0) {
        report_error(routine, 3);
        return;
    }
    if (k < 0) {
        report_error(routine, 4);
        return;
    }
    const bool no_trans = to_upper(trans) == 'N';
    if (lda < std::max(1, no_trans ? n : k)) {
        report_error(routine, 7);
        return;
    }
    if (ldc < std::max(1, n)) {
        report_error(routine, 10);
        return;
    }
    if (n == 0) {
        return;
    }
    if ((alpha == Scalar(0) || k == 0) && beta == Scalar(1)) {
        return;
    }
    for (int j = 0; j < n; ++j) {
        const int ibegin = to_upper(uplo) == 'U' ? 0 : j;
        const int iend = to_upper(uplo) == 'U' ? j + 1 : n;
        for (int i = ibegin; i < iend; ++i) {
            T sum = T(0);
            if (alpha != Scalar(0)) {
                for (int l = 0; l < k; ++l) {
                    if constexpr (Hermitian) {
                        sum += op_value(a, lda, trans, i, l) *
                               conj_val(op_value(a, lda, trans, j, l));
                    } else {
                        sum += op_value(a, lda, trans, i, l) *
                               op_value(a, lda, trans, j, l);
                    }
                }
            }
            auto &value = c[idx(i, j, ldc)];
            value = static_cast<T>(alpha) * sum +
                    (beta == Scalar(0) ? T(0) : static_cast<T>(beta) * value);
            if constexpr (Hermitian) {
                if (i == j) {
                    value = real_diagonal(value);
                }
            }
        }
    }
}

template <typename T, typename ScalarAlpha, typename ScalarBeta, bool Hermitian>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void syr2k_impl(const char *routine, char uplo, char trans, int n, int k, ScalarAlpha alpha,
                const T *a, int lda, const T *b, int ldb, ScalarBeta beta, T *c, int ldc) {
    if (!valid_uplo(uplo)) {
        report_error(routine, 1);
        return;
    }
    if (!valid_rank_trans(trans, Hermitian)) {
        report_error(routine, 2);
        return;
    }
    if (n < 0) {
        report_error(routine, 3);
        return;
    }
    if (k < 0) {
        report_error(routine, 4);
        return;
    }
    const bool no_trans = to_upper(trans) == 'N';
    const int min_lda = std::max(1, no_trans ? n : k);
    if (lda < min_lda) {
        report_error(routine, 7);
        return;
    }
    if (ldb < min_lda) {
        report_error(routine, 9);
        return;
    }
    if (ldc < std::max(1, n)) {
        report_error(routine, 12);
        return;
    }
    if (n == 0) {
        return;
    }
    if ((alpha == ScalarAlpha(0) || k == 0) && beta == ScalarBeta(1)) {
        return;
    }
    for (int j = 0; j < n; ++j) {
        const int ibegin = to_upper(uplo) == 'U' ? 0 : j;
        const int iend = to_upper(uplo) == 'U' ? j + 1 : n;
        for (int i = ibegin; i < iend; ++i) {
            T sum = T(0);
            if (alpha != ScalarAlpha(0)) {
                for (int l = 0; l < k; ++l) {
                    const T ai = op_value(a, lda, trans, i, l);
                    const T aj = op_value(a, lda, trans, j, l);
                    const T bi = op_value(b, ldb, trans, i, l);
                    const T bj = op_value(b, ldb, trans, j, l);
                    if constexpr (Hermitian) {
                        const T complex_alpha = static_cast<T>(alpha);
                        sum += complex_alpha * ai * conj_val(bj) +
                               conj_val(complex_alpha) * bi * conj_val(aj);
                    } else {
                        sum += ai * bj + bi * aj;
                    }
                }
            }
            auto &value = c[idx(i, j, ldc)];
            if constexpr (Hermitian) {
                value = sum + (beta == ScalarBeta(0) ? T(0) : static_cast<T>(beta) * value);
            } else {
                value = static_cast<T>(alpha) * sum +
                        (beta == ScalarBeta(0) ? T(0) : static_cast<T>(beta) * value);
            }
            if constexpr (Hermitian) {
                if (i == j) {
                    value = real_diagonal(value);
                }
            }
        }
    }
}

template <typename T>
T triangular_value(const T *a, int lda, char uplo, char trans, char diag, int row, int col) {
    const char ul = to_upper(uplo);
    const char tr = to_upper(trans);
    if (row == col && to_upper(diag) == 'U') {
        return T(1);
    }
    const int source_row = tr == 'N' ? row : col;
    const int source_col = tr == 'N' ? col : row;
    if ((ul == 'U' && source_row > source_col) || (ul == 'L' && source_row < source_col)) {
        return T(0);
    }
    const T value = a[idx(source_row, source_col, lda)];
    return tr == 'C' ? conj_val(value) : value;
}

int triangular_info(char side, char uplo, char trans, char diag, int m, int n, int lda, int ldb) {
    if (!valid_side(side)) {
        return 1;
    }
    if (!valid_uplo(uplo)) {
        return 2;
    }
    if (!valid_trans(trans)) {
        return 3;
    }
    if (!valid_diag(diag)) {
        return 4;
    }
    if (m < 0) {
        return 5;
    }
    if (n < 0) {
        return 6;
    }
    const int order = to_upper(side) == 'L' ? m : n;
    if (lda < std::max(1, order)) {
        return 9;
    }
    if (ldb < std::max(1, m)) {
        return 11;
    }
    return 0;
}

template <typename T>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void trmm_impl(const char *routine, char side, char uplo, char trans, char diag, int m, int n,
               T alpha, const T *a, int lda, T *b, int ldb) {
    const int info = triangular_info(side, uplo, trans, diag, m, n, lda, ldb);
    if (info != 0) {
        report_error(routine, info);
        return;
    }
    if (m == 0 || n == 0) {
        return;
    }
    if (alpha == T(0)) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                b[idx(i, j, ldb)] = T(0);
            }
        }
        return;
    }
    const bool left = to_upper(side) == 'L';
    std::vector<T> result(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            T sum = T(0);
            const int order = left ? m : n;
            for (int l = 0; l < order; ++l) {
                if (left) {
                    sum += triangular_value(a, lda, uplo, trans, diag, i, l) * b[idx(l, j, ldb)];
                } else {
                    sum += b[idx(i, l, ldb)] * triangular_value(a, lda, uplo, trans, diag, l, j);
                }
            }
            result[static_cast<std::size_t>(idx(i, j, m))] = alpha * sum;
        }
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            b[idx(i, j, ldb)] = result[static_cast<std::size_t>(idx(i, j, m))];
        }
    }
}

template <typename T>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void trsm_impl(const char *routine, char side, char uplo, char trans, char diag, int m, int n,
               T alpha, const T *a, int lda, T *b, int ldb) {
    const int info = triangular_info(side, uplo, trans, diag, m, n, lda, ldb);
    if (info != 0) {
        report_error(routine, info);
        return;
    }
    if (m == 0 || n == 0) {
        return;
    }
    if (alpha == T(0)) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                b[idx(i, j, ldb)] = T(0);
            }
        }
        return;
    }
    const bool left = to_upper(side) == 'L';
    const int order = left ? m : n;
    const bool upper = (to_upper(trans) == 'N') == (to_upper(uplo) == 'U');
    const bool backward = left ? upper : !upper;
    for (int rhs = 0; rhs < (left ? n : m); ++rhs) {
        auto value_at = [&](int i) -> T & {
            return left ? b[idx(i, rhs, ldb)] : b[idx(rhs, i, ldb)];
        };
        for (int i = 0; i < order; ++i) {
            value_at(i) *= alpha;
        }
        for (int step = 0; step < order; ++step) {
            const int i = backward ? order - 1 - step : step;
            T value = value_at(i);
            if (backward) {
                for (int j = i + 1; j < order; ++j) {
                    const T coeff = left ? triangular_value(a, lda, uplo, trans, diag, i, j)
                                         : triangular_value(a, lda, uplo, trans, diag, j, i);
                    value -= coeff * value_at(j);
                }
            } else {
                for (int j = 0; j < i; ++j) {
                    const T coeff = left ? triangular_value(a, lda, uplo, trans, diag, i, j)
                                         : triangular_value(a, lda, uplo, trans, diag, j, i);
                    value -= coeff * value_at(j);
                }
            }
            if (to_upper(diag) != 'U') {
                value /= triangular_value(a, lda, uplo, trans, diag, i, i);
            }
            value_at(i) = value;
        }
    }
}

} // namespace

// Macro arguments used as parameter types cannot be parenthesized.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define THEBLAS_GEMM(NAME, ROUTINE, TYPE)                                                          \
    void NAME(char transa, char transb, int m, int n, int k, TYPE alpha, const TYPE *a, int lda,   \
              const TYPE *b, int ldb, TYPE beta, TYPE *c, int ldc) {                               \
        gemm_impl(ROUTINE, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);          \
    }
#define THEBLAS_SYMM(NAME, ROUTINE, TYPE, HERM)                                                    \
    void NAME(char side, char uplo, int m, int n, TYPE alpha, const TYPE *a, int lda,              \
              const TYPE *b, int ldb, TYPE beta, TYPE *c, int ldc) {                               \
        symm_impl<TYPE, HERM>(ROUTINE, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);     \
    }
#define THEBLAS_SYRK(NAME, ROUTINE, TYPE, SCALAR, HERM)                                            \
    void NAME(char uplo, char trans, int n, int k, SCALAR alpha, const TYPE *a, int lda,           \
              SCALAR beta, TYPE *c, int ldc) {                                                     \
        syrk_impl<TYPE, SCALAR, HERM>(ROUTINE, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);    \
    }
#define THEBLAS_SYR2K(NAME, ROUTINE, TYPE, ALPHA, BETA, HERM)                                      \
    void NAME(char uplo, char trans, int n, int k, ALPHA alpha, const TYPE *a, int lda,            \
              const TYPE *b, int ldb, BETA beta, TYPE *c, int ldc) {                               \
        syr2k_impl<TYPE, ALPHA, BETA, HERM>(ROUTINE, uplo, trans, n, k, alpha, a, lda, b, ldb,     \
                                             beta, c, ldc);                                         \
    }
#define THEBLAS_TRMM(NAME, ROUTINE, TYPE)                                                          \
    void NAME(char side, char uplo, char trans, char diag, int m, int n, TYPE alpha,               \
              const TYPE *a, int lda, TYPE *b, int ldb) {                                          \
        trmm_impl(ROUTINE, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);                  \
    }
#define THEBLAS_TRSM(NAME, ROUTINE, TYPE)                                                          \
    void NAME(char side, char uplo, char trans, char diag, int m, int n, TYPE alpha,               \
              const TYPE *a, int lda, TYPE *b, int ldb) {                                          \
        trsm_impl(ROUTINE, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);                  \
    }
// NOLINTEND(bugprone-macro-parentheses)

THEBLAS_GEMM(sgemm, "SGEMM", float)
THEBLAS_GEMM(dgemm, "DGEMM", double)
THEBLAS_GEMM(cgemm, "CGEMM", std::complex<float>)
THEBLAS_GEMM(zgemm, "ZGEMM", std::complex<double>)
THEBLAS_SYMM(ssymm, "SSYMM", float, false)
THEBLAS_SYMM(dsymm, "DSYMM", double, false)
THEBLAS_SYMM(csymm, "CSYMM", std::complex<float>, false)
THEBLAS_SYMM(zsymm, "ZSYMM", std::complex<double>, false)
THEBLAS_SYMM(chemm, "CHEMM", std::complex<float>, true)
THEBLAS_SYMM(zhemm, "ZHEMM", std::complex<double>, true)
THEBLAS_SYRK(ssyrk, "SSYRK", float, float, false)
THEBLAS_SYRK(dsyrk, "DSYRK", double, double, false)
THEBLAS_SYRK(csyrk, "CSYRK", std::complex<float>, std::complex<float>, false)
THEBLAS_SYRK(zsyrk, "ZSYRK", std::complex<double>, std::complex<double>, false)
THEBLAS_SYRK(cherk, "CHERK", std::complex<float>, float, true)
THEBLAS_SYRK(zherk, "ZHERK", std::complex<double>, double, true)
THEBLAS_SYR2K(ssyr2k, "SSYR2K", float, float, float, false)
THEBLAS_SYR2K(dsyr2k, "DSYR2K", double, double, double, false)
THEBLAS_SYR2K(csyr2k, "CSYR2K", std::complex<float>, std::complex<float>, std::complex<float>,
              false)
THEBLAS_SYR2K(zsyr2k, "ZSYR2K", std::complex<double>, std::complex<double>, std::complex<double>,
              false)
THEBLAS_SYR2K(cher2k, "CHER2K", std::complex<float>, std::complex<float>, float, true)
THEBLAS_SYR2K(zher2k, "ZHER2K", std::complex<double>, std::complex<double>, double, true)
THEBLAS_TRMM(strmm, "STRMM", float)
THEBLAS_TRMM(dtrmm, "DTRMM", double)
THEBLAS_TRMM(ctrmm, "CTRMM", std::complex<float>)
THEBLAS_TRMM(ztrmm, "ZTRMM", std::complex<double>)
THEBLAS_TRSM(strsm, "STRSM", float)
THEBLAS_TRSM(dtrsm, "DTRSM", double)
THEBLAS_TRSM(ctrsm, "CTRSM", std::complex<float>)
THEBLAS_TRSM(ztrsm, "ZTRSM", std::complex<double>)

#undef THEBLAS_GEMM
#undef THEBLAS_SYMM
#undef THEBLAS_SYRK
#undef THEBLAS_SYR2K
#undef THEBLAS_TRMM
#undef THEBLAS_TRSM

} // namespace theblas
// NOLINTEND(bugprone-easily-swappable-parameters)
