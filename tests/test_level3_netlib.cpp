#include "theblas/theblas.h"

#include "test_support.hpp"

#include <cblas.h>

#include <array>
#include <cassert>
#include <complex>
#include <initializer_list>

namespace {

using theblas::test::almost_equal;

template <typename T> struct scalar_value {
    static T make(double real, double) { return static_cast<T>(real); }
};

template <typename T> struct scalar_value<std::complex<T>> {
    static std::complex<T> make(double real, double imag) {
        return {static_cast<T>(real), static_cast<T>(imag)};
    }
};

template <typename T> T value(double real, double imag = 0.0) {
    return scalar_value<T>::make(real, imag);
}

template <typename T, std::size_t N>
void expect_equal(const std::array<T, N> &lhs, const std::array<T, N> &rhs) {
    for (std::size_t i = 0; i < N; ++i)
        assert(almost_equal(lhs[i], rhs[i]));
}

CBLAS_TRANSPOSE transpose(char trans) {
    if (trans == 'N')
        return CblasNoTrans;
    if (trans == 'T')
        return CblasTrans;
    return CblasConjTrans;
}

CBLAS_UPLO triangle(char uplo) {
    return uplo == 'U' ? CblasUpper : CblasLower;
}

CBLAS_SIDE matrix_side(char side) {
    return side == 'L' ? CblasLeft : CblasRight;
}

CBLAS_DIAG diagonal(char diag) {
    return diag == 'U' ? CblasUnit : CblasNonUnit;
}

template <typename T, typename OurCall, typename NetlibCall>
void compare_gemm(OurCall ours, NetlibCall netlib, std::initializer_list<char> trans) {
    const std::array<T, 6> a = {value<T>(1, 0.5),  value<T>(2, -1),   T(91),
                                value<T>(3, 0.25), value<T>(4, -0.5), T(92)};
    const std::array<T, 6> b = {value<T>(2, 0.75), value<T>(1, -0.25), T(93),
                                value<T>(0, 0.5),  value<T>(3, -1),    T(94)};
    for (const char ta : trans) {
        for (const char tb : trans) {
            std::array<T, 6> c1 = {value<T>(1, 0.5),  value<T>(2, -0.5),  T(95),
                                   value<T>(3, 0.25), value<T>(4, -0.25), T(96)};
            auto c2 = c1;
            ours(ta, tb, a.data(), b.data(), c1.data());
            netlib(ta, tb, a.data(), b.data(), c2.data());
            expect_equal(c1, c2);
        }
    }
}

template <typename T, typename OurCall, typename NetlibCall>
void compare_side_matrix(OurCall ours, NetlibCall netlib) {
    const std::array<T, 4> a = {value<T>(2), value<T>(0), value<T>(1, 1), value<T>(3)};
    const std::array<T, 6> b = {value<T>(1, 0.5), value<T>(2, -0.25), T(90),
                                value<T>(3, 1),   value<T>(4, -0.5),  T(91)};
    for (const char side : {'L', 'R'}) {
        for (const char uplo : {'U', 'L'}) {
            std::array<T, 6> c1 = {value<T>(2, 0.5),  value<T>(3, -0.5),  T(92),
                                   value<T>(4, 0.25), value<T>(5, -0.25), T(93)};
            auto c2 = c1;
            ours(side, uplo, a.data(), b.data(), c1.data());
            netlib(side, uplo, a.data(), b.data(), c2.data());
            expect_equal(c1, c2);
        }
    }
}

template <typename T, typename OurCall, typename NetlibCall>
void compare_rank_k(OurCall ours, NetlibCall netlib, std::initializer_list<char> trans) {
    const std::array<T, 6> a = {value<T>(1, 0.5),  value<T>(2, -1),   T(90),
                                value<T>(3, 0.25), value<T>(4, -0.5), T(91)};
    for (const char ul : {'U', 'L'}) {
        for (const char tr : trans) {
            std::array<T, 6> c1 = {value<T>(1, 0.5),  value<T>(2, -0.5),  T(92),
                                   value<T>(3, 0.25), value<T>(4, -0.25), T(93)};
            auto c2 = c1;
            ours(ul, tr, a.data(), c1.data());
            netlib(ul, tr, a.data(), c2.data());
            expect_equal(c1, c2);
        }
    }
}

template <typename T, typename OurCall, typename NetlibCall>
void compare_rank_2k(OurCall ours, NetlibCall netlib, std::initializer_list<char> trans) {
    const std::array<T, 6> a = {value<T>(1, 0.5),  value<T>(2, -1),   T(90),
                                value<T>(3, 0.25), value<T>(4, -0.5), T(91)};
    const std::array<T, 6> b = {value<T>(2, -0.75), value<T>(3, 0.5), T(92),
                                value<T>(1, 0.25),  value<T>(2, -1),  T(93)};
    for (const char ul : {'U', 'L'}) {
        for (const char tr : trans) {
            std::array<T, 6> c1 = {value<T>(1, 0.5),  value<T>(2, -0.5),  T(94),
                                   value<T>(3, 0.25), value<T>(4, -0.25), T(95)};
            auto c2 = c1;
            ours(ul, tr, a.data(), b.data(), c1.data());
            netlib(ul, tr, a.data(), b.data(), c2.data());
            expect_equal(c1, c2);
        }
    }
}

template <typename T, typename OurCall, typename NetlibCall>
void compare_triangular(OurCall ours, NetlibCall netlib, std::initializer_list<char> trans) {
    const std::array<T, 4> a = {value<T>(2, 0.5), value<T>(3, -1), value<T>(4, 1.5),
                                value<T>(5, -0.25)};
    for (const char side : {'L', 'R'}) {
        for (const char ul : {'U', 'L'}) {
            for (const char tr : trans) {
                for (const char diag : {'N', 'U'}) {
                    std::array<T, 6> b1 = {value<T>(1, 0.5),  value<T>(2, -0.5),  T(92),
                                           value<T>(3, 0.25), value<T>(4, -0.25), T(93)};
                    auto b2 = b1;
                    ours(side, ul, tr, diag, a.data(), b1.data());
                    netlib(side, ul, tr, diag, a.data(), b2.data());
                    expect_equal(b1, b2);
                }
            }
        }
    }
}

} // namespace

void run_level3_netlib_tests() {
    using cf = std::complex<float>;
    using cd = std::complex<double>;

    compare_gemm<float>(
        [](char ta, char tb, const float *a, const float *b, float *c) {
            theblas::sgemm(ta, tb, 2, 2, 2, 1.25F, a, 3, b, 3, 0.5F, c, 3);
        },
        [](char ta, char tb, const float *a, const float *b, float *c) {
            cblas_sgemm(CblasColMajor, transpose(ta), transpose(tb), 2, 2, 2, 1.25F, a, 3, b, 3,
                        0.5F, c, 3);
        },
        {'N', 'T'});
    compare_gemm<double>(
        [](char ta, char tb, const double *a, const double *b, double *c) {
            theblas::dgemm(ta, tb, 2, 2, 2, 1.25, a, 3, b, 3, 0.5, c, 3);
        },
        [](char ta, char tb, const double *a, const double *b, double *c) {
            cblas_dgemm(CblasColMajor, transpose(ta), transpose(tb), 2, 2, 2, 1.25, a, 3, b, 3, 0.5,
                        c, 3);
        },
        {'N', 'T'});
    compare_gemm<cf>(
        [](char ta, char tb, const cf *a, const cf *b, cf *c) {
            theblas::cgemm(ta, tb, 2, 2, 2, cf(1.25F, -0.5F), a, 3, b, 3, cf(0.5F, 0.25F), c, 3);
        },
        [](char ta, char tb, const cf *a, const cf *b, cf *c) {
            const cf alpha(1.25F, -0.5F), beta(0.5F, 0.25F);
            cblas_cgemm(CblasColMajor, transpose(ta), transpose(tb), 2, 2, 2, &alpha, a, 3, b, 3,
                        &beta, c, 3);
        },
        {'N', 'T', 'C'});
    compare_gemm<cd>(
        [](char ta, char tb, const cd *a, const cd *b, cd *c) {
            theblas::zgemm(ta, tb, 2, 2, 2, cd(1.25, -0.5), a, 3, b, 3, cd(0.5, 0.25), c, 3);
        },
        [](char ta, char tb, const cd *a, const cd *b, cd *c) {
            const cd alpha(1.25, -0.5), beta(0.5, 0.25);
            cblas_zgemm(CblasColMajor, transpose(ta), transpose(tb), 2, 2, 2, &alpha, a, 3, b, 3,
                        &beta, c, 3);
        },
        {'N', 'T', 'C'});

    compare_side_matrix<float>(
        [](char side, char ul, const float *a, const float *b, float *c) {
            theblas::ssymm(side, ul, 2, 2, 1.25F, a, 2, b, 3, 0.5F, c, 3);
        },
        [](char side, char ul, const float *a, const float *b, float *c) {
            cblas_ssymm(CblasColMajor, matrix_side(side), triangle(ul), 2, 2, 1.25F, a, 2, b, 3,
                        0.5F, c, 3);
        });
    compare_side_matrix<double>(
        [](char side, char ul, const double *a, const double *b, double *c) {
            theblas::dsymm(side, ul, 2, 2, 1.25, a, 2, b, 3, 0.5, c, 3);
        },
        [](char side, char ul, const double *a, const double *b, double *c) {
            cblas_dsymm(CblasColMajor, matrix_side(side), triangle(ul), 2, 2, 1.25, a, 2, b, 3, 0.5,
                        c, 3);
        });
    compare_side_matrix<cf>(
        [](char side, char ul, const cf *a, const cf *b, cf *c) {
            theblas::csymm(side, ul, 2, 2, cf(1.25F, -0.5F), a, 2, b, 3, cf(0.5F, 0.25F), c, 3);
        },
        [](char side, char ul, const cf *a, const cf *b, cf *c) {
            const cf alpha(1.25F, -0.5F), beta(0.5F, 0.25F);
            cblas_csymm(CblasColMajor, matrix_side(side), triangle(ul), 2, 2, &alpha, a, 2, b, 3,
                        &beta, c, 3);
        });
    compare_side_matrix<cd>(
        [](char side, char ul, const cd *a, const cd *b, cd *c) {
            theblas::zsymm(side, ul, 2, 2, cd(1.25, -0.5), a, 2, b, 3, cd(0.5, 0.25), c, 3);
        },
        [](char side, char ul, const cd *a, const cd *b, cd *c) {
            const cd alpha(1.25, -0.5), beta(0.5, 0.25);
            cblas_zsymm(CblasColMajor, matrix_side(side), triangle(ul), 2, 2, &alpha, a, 2, b, 3,
                        &beta, c, 3);
        });
    compare_side_matrix<cf>(
        [](char side, char ul, const cf *a, const cf *b, cf *c) {
            theblas::chemm(side, ul, 2, 2, cf(1.25F, -0.5F), a, 2, b, 3, cf(0.5F, 0.25F), c, 3);
        },
        [](char side, char ul, const cf *a, const cf *b, cf *c) {
            const cf alpha(1.25F, -0.5F), beta(0.5F, 0.25F);
            cblas_chemm(CblasColMajor, matrix_side(side), triangle(ul), 2, 2, &alpha, a, 2, b, 3,
                        &beta, c, 3);
        });
    compare_side_matrix<cd>(
        [](char side, char ul, const cd *a, const cd *b, cd *c) {
            theblas::zhemm(side, ul, 2, 2, cd(1.25, -0.5), a, 2, b, 3, cd(0.5, 0.25), c, 3);
        },
        [](char side, char ul, const cd *a, const cd *b, cd *c) {
            const cd alpha(1.25, -0.5), beta(0.5, 0.25);
            cblas_zhemm(CblasColMajor, matrix_side(side), triangle(ul), 2, 2, &alpha, a, 2, b, 3,
                        &beta, c, 3);
        });

    compare_rank_k<float>(
        [](char ul, char tr, const float *a, float *c) {
            theblas::ssyrk(ul, tr, 2, 2, 1.25F, a, tr == 'N' ? 3 : 3, 0.5F, c, 3);
        },
        [](char ul, char tr, const float *a, float *c) {
            cblas_ssyrk(CblasColMajor, triangle(ul), transpose(tr), 2, 2, 1.25F, a, 3, 0.5F, c, 3);
        },
        {'N', 'T'});
    compare_rank_k<double>([](char ul, char tr, const double *a,
                              double *c) { theblas::dsyrk(ul, tr, 2, 2, 1.25, a, 3, 0.5, c, 3); },
                           [](char ul, char tr, const double *a, double *c) {
                               cblas_dsyrk(CblasColMajor, triangle(ul), transpose(tr), 2, 2, 1.25,
                                           a, 3, 0.5, c, 3);
                           },
                           {'N', 'T'});
    compare_rank_k<cf>(
        [](char ul, char tr, const cf *a, cf *c) {
            theblas::csyrk(ul, tr, 2, 2, cf(1.25F, -0.5F), a, 3, cf(0.5F, 0.25F), c, 3);
        },
        [](char ul, char tr, const cf *a, cf *c) {
            const cf alpha(1.25F, -0.5F), beta(0.5F, 0.25F);
            cblas_csyrk(CblasColMajor, triangle(ul), transpose(tr), 2, 2, &alpha, a, 3, &beta, c,
                        3);
        },
        {'N', 'T'});
    compare_rank_k<cd>(
        [](char ul, char tr, const cd *a, cd *c) {
            theblas::zsyrk(ul, tr, 2, 2, cd(1.25, -0.5), a, 3, cd(0.5, 0.25), c, 3);
        },
        [](char ul, char tr, const cd *a, cd *c) {
            const cd alpha(1.25, -0.5), beta(0.5, 0.25);
            cblas_zsyrk(CblasColMajor, triangle(ul), transpose(tr), 2, 2, &alpha, a, 3, &beta, c,
                        3);
        },
        {'N', 'T'});
    compare_rank_k<cf>([](char ul, char tr, const cf *a,
                          cf *c) { theblas::cherk(ul, tr, 2, 2, 1.25F, a, 3, 0.5F, c, 3); },
                       [](char ul, char tr, const cf *a, cf *c) {
                           cblas_cherk(CblasColMajor, triangle(ul), transpose(tr), 2, 2, 1.25F, a,
                                       3, 0.5F, c, 3);
                       },
                       {'N', 'C'});
    compare_rank_k<cd>([](char ul, char tr, const cd *a,
                          cd *c) { theblas::zherk(ul, tr, 2, 2, 1.25, a, 3, 0.5, c, 3); },
                       [](char ul, char tr, const cd *a, cd *c) {
                           cblas_zherk(CblasColMajor, triangle(ul), transpose(tr), 2, 2, 1.25, a, 3,
                                       0.5, c, 3);
                       },
                       {'N', 'C'});

    compare_rank_2k<float>(
        [](char ul, char tr, const float *a, const float *b, float *c) {
            theblas::ssyr2k(ul, tr, 2, 2, 1.25F, a, 3, b, 3, 0.5F, c, 3);
        },
        [](char ul, char tr, const float *a, const float *b, float *c) {
            cblas_ssyr2k(CblasColMajor, triangle(ul), transpose(tr), 2, 2, 1.25F, a, 3, b, 3, 0.5F,
                         c, 3);
        },
        {'N', 'T'});
    compare_rank_2k<double>(
        [](char ul, char tr, const double *a, const double *b, double *c) {
            theblas::dsyr2k(ul, tr, 2, 2, 1.25, a, 3, b, 3, 0.5, c, 3);
        },
        [](char ul, char tr, const double *a, const double *b, double *c) {
            cblas_dsyr2k(CblasColMajor, triangle(ul), transpose(tr), 2, 2, 1.25, a, 3, b, 3, 0.5, c,
                         3);
        },
        {'N', 'T'});
    compare_rank_2k<cf>(
        [](char ul, char tr, const cf *a, const cf *b, cf *c) {
            theblas::csyr2k(ul, tr, 2, 2, cf(1.25F, -0.5F), a, 3, b, 3, cf(0.5F, 0.25F), c, 3);
        },
        [](char ul, char tr, const cf *a, const cf *b, cf *c) {
            const cf alpha(1.25F, -0.5F), beta(0.5F, 0.25F);
            cblas_csyr2k(CblasColMajor, triangle(ul), transpose(tr), 2, 2, &alpha, a, 3, b, 3,
                         &beta, c, 3);
        },
        {'N', 'T'});
    compare_rank_2k<cd>(
        [](char ul, char tr, const cd *a, const cd *b, cd *c) {
            theblas::zsyr2k(ul, tr, 2, 2, cd(1.25, -0.5), a, 3, b, 3, cd(0.5, 0.25), c, 3);
        },
        [](char ul, char tr, const cd *a, const cd *b, cd *c) {
            const cd alpha(1.25, -0.5), beta(0.5, 0.25);
            cblas_zsyr2k(CblasColMajor, triangle(ul), transpose(tr), 2, 2, &alpha, a, 3, b, 3,
                         &beta, c, 3);
        },
        {'N', 'T'});
    compare_rank_2k<cf>(
        [](char ul, char tr, const cf *a, const cf *b, cf *c) {
            theblas::cher2k(ul, tr, 2, 2, cf(1.25F, -0.5F), a, 3, b, 3, 0.5F, c, 3);
        },
        [](char ul, char tr, const cf *a, const cf *b, cf *c) {
            const cf alpha(1.25F, -0.5F);
            cblas_cher2k(CblasColMajor, triangle(ul), transpose(tr), 2, 2, &alpha, a, 3, b, 3, 0.5F,
                         c, 3);
        },
        {'N', 'C'});
    compare_rank_2k<cd>(
        [](char ul, char tr, const cd *a, const cd *b, cd *c) {
            theblas::zher2k(ul, tr, 2, 2, cd(1.25, -0.5), a, 3, b, 3, 0.5, c, 3);
        },
        [](char ul, char tr, const cd *a, const cd *b, cd *c) {
            const cd alpha(1.25, -0.5);
            cblas_zher2k(CblasColMajor, triangle(ul), transpose(tr), 2, 2, &alpha, a, 3, b, 3, 0.5,
                         c, 3);
        },
        {'N', 'C'});

    compare_triangular<float>(
        [](char side, char ul, char tr, char dg, const float *a, float *b) {
            theblas::strmm(side, ul, tr, dg, 2, 2, 1.25F, a, 2, b, 3);
        },
        [](char side, char ul, char tr, char dg, const float *a, float *b) {
            cblas_strmm(CblasColMajor, matrix_side(side), triangle(ul), transpose(tr), diagonal(dg),
                        2, 2, 1.25F, a, 2, b, 3);
        },
        {'N', 'T'});
    compare_triangular<double>(
        [](char side, char ul, char tr, char dg, const double *a, double *b) {
            theblas::dtrmm(side, ul, tr, dg, 2, 2, 1.25, a, 2, b, 3);
        },
        [](char side, char ul, char tr, char dg, const double *a, double *b) {
            cblas_dtrmm(CblasColMajor, matrix_side(side), triangle(ul), transpose(tr), diagonal(dg),
                        2, 2, 1.25, a, 2, b, 3);
        },
        {'N', 'T'});
    compare_triangular<cf>(
        [](char side, char ul, char tr, char dg, const cf *a, cf *b) {
            theblas::ctrmm(side, ul, tr, dg, 2, 2, cf(1.25F, -0.5F), a, 2, b, 3);
        },
        [](char side, char ul, char tr, char dg, const cf *a, cf *b) {
            const cf alpha(1.25F, -0.5F);
            cblas_ctrmm(CblasColMajor, matrix_side(side), triangle(ul), transpose(tr), diagonal(dg),
                        2, 2, &alpha, a, 2, b, 3);
        },
        {'N', 'T', 'C'});
    compare_triangular<cd>(
        [](char side, char ul, char tr, char dg, const cd *a, cd *b) {
            theblas::ztrmm(side, ul, tr, dg, 2, 2, cd(1.25, -0.5), a, 2, b, 3);
        },
        [](char side, char ul, char tr, char dg, const cd *a, cd *b) {
            const cd alpha(1.25, -0.5);
            cblas_ztrmm(CblasColMajor, matrix_side(side), triangle(ul), transpose(tr), diagonal(dg),
                        2, 2, &alpha, a, 2, b, 3);
        },
        {'N', 'T', 'C'});

    compare_triangular<float>(
        [](char side, char ul, char tr, char dg, const float *a, float *b) {
            theblas::strsm(side, ul, tr, dg, 2, 2, 1.25F, a, 2, b, 3);
        },
        [](char side, char ul, char tr, char dg, const float *a, float *b) {
            cblas_strsm(CblasColMajor, matrix_side(side), triangle(ul), transpose(tr), diagonal(dg),
                        2, 2, 1.25F, a, 2, b, 3);
        },
        {'N', 'T'});
    compare_triangular<double>(
        [](char side, char ul, char tr, char dg, const double *a, double *b) {
            theblas::dtrsm(side, ul, tr, dg, 2, 2, 1.25, a, 2, b, 3);
        },
        [](char side, char ul, char tr, char dg, const double *a, double *b) {
            cblas_dtrsm(CblasColMajor, matrix_side(side), triangle(ul), transpose(tr), diagonal(dg),
                        2, 2, 1.25, a, 2, b, 3);
        },
        {'N', 'T'});
    compare_triangular<cf>(
        [](char side, char ul, char tr, char dg, const cf *a, cf *b) {
            theblas::ctrsm(side, ul, tr, dg, 2, 2, cf(1.25F, -0.5F), a, 2, b, 3);
        },
        [](char side, char ul, char tr, char dg, const cf *a, cf *b) {
            const cf alpha(1.25F, -0.5F);
            cblas_ctrsm(CblasColMajor, matrix_side(side), triangle(ul), transpose(tr), diagonal(dg),
                        2, 2, &alpha, a, 2, b, 3);
        },
        {'N', 'T', 'C'});
    compare_triangular<cd>(
        [](char side, char ul, char tr, char dg, const cd *a, cd *b) {
            theblas::ztrsm(side, ul, tr, dg, 2, 2, cd(1.25, -0.5), a, 2, b, 3);
        },
        [](char side, char ul, char tr, char dg, const cd *a, cd *b) {
            const cd alpha(1.25, -0.5);
            cblas_ztrsm(CblasColMajor, matrix_side(side), triangle(ul), transpose(tr), diagonal(dg),
                        2, 2, &alpha, a, 2, b, 3);
        },
        {'N', 'T', 'C'});
}
