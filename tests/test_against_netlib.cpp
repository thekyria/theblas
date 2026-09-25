#include "theblas/theblas.h"

#include "test_support.hpp"

#include <cblas.h>

#include <array>
#include <cassert>
#include <complex>

namespace {

using theblas::test::almost_equal;

CBLAS_TRANSPOSE as_trans(char value) {
    switch (value) {
    case 'N':
    case 'n':
        return CblasNoTrans;
    case 'T':
    case 't':
        return CblasTrans;
    default:
        return CblasConjTrans;
    }
}

CBLAS_UPLO as_uplo(char value) {
    return (value == 'U' || value == 'u') ? CblasUpper : CblasLower;
}

CBLAS_DIAG as_diag(char value) {
    return (value == 'U' || value == 'u') ? CblasUnit : CblasNonUnit;
}

template <typename T, std::size_t N>
void expect_equal(const std::array<T, N> &lhs, const std::array<T, N> &rhs) {
    for (std::size_t i = 0; i < N; ++i) {
        assert(almost_equal(lhs[i], rhs[i]));
    }
}

template <typename T> void expect_equal(const T &lhs, const T &rhs) {
    assert(almost_equal(lhs, rhs));
}

} // namespace

int main() {
    {
        std::array<float, 5> x1 = {1.0F, 9.0F, 2.0F, 8.0F, 3.0F};
        std::array<float, 5> y1 = {4.0F, 7.0F, 5.0F, 6.0F, 6.0F};
        auto x2 = x1;
        auto y2 = y1;
        theblas::sswap(3, x1.data(), 2, y1.data(), 2);
        cblas_sswap(3, x2.data(), 2, y2.data(), 2);
        expect_equal(x1, x2);
        expect_equal(y1, y2);
    }
    {
        std::array<double, 5> x = {1.0, 100.0, 2.0, 90.0, 3.0};
        std::array<double, 3> y1 = {0.0, 0.0, 0.0};
        auto y2 = y1;
        theblas::dcopy(3, x.data(), -2, y1.data(), 1);
        cblas_dcopy(3, x.data() + 4, -2, y2.data(), 1);
        expect_equal(y1, y2);
    }
    {
        std::array<std::complex<float>, 2> x = {{{1.0F, 1.0F}, {2.0F, -1.0F}}};
        std::array<std::complex<float>, 2> y1 = {{{3.0F, 0.0F}, {1.0F, 1.0F}}};
        auto y2 = y1;
        const std::complex<float> alpha(2.0F, 0.0F);
        theblas::caxpy(2, alpha, x.data(), 1, y1.data(), 1);
        cblas_caxpy(2, &alpha, x.data(), 1, y2.data(), 1);
        expect_equal(y1, y2);
    }
    {
        std::array<std::complex<double>, 3> x1 = {{{1.0, 2.0}, {3.0, -1.0}, {-2.0, 1.0}}};
        auto x2 = x1;
        const std::complex<double> alpha(0.5, -1.0);
        theblas::zscal(3, alpha, x1.data(), 1);
        cblas_zscal(3, &alpha, x2.data(), 1);
        expect_equal(x1, x2);
    }
    {
        std::array<std::complex<float>, 2> x1 = {{{1.0F, -1.0F}, {2.0F, 3.0F}}};
        auto x2 = x1;
        theblas::csscal(2, 0.25F, x1.data(), 1);
        cblas_csscal(2, 0.25F, x2.data(), 1);
        expect_equal(x1, x2);
    }
    {
        std::array<std::complex<double>, 2> x1 = {{{1.0, -1.0}, {2.0, 3.0}}};
        auto x2 = x1;
        theblas::zdscal(2, 0.25, x1.data(), 1);
        cblas_zdscal(2, 0.25, x2.data(), 1);
        expect_equal(x1, x2);
    }
    {
        std::array<float, 3> x = {1.0F, 2.0F, 3.0F};
        std::array<float, 3> y = {4.0F, 5.0F, 6.0F};
        expect_equal(theblas::sdot(3, x.data(), 1, y.data(), 1),
                     cblas_sdot(3, x.data(), 1, y.data(), 1));
        assert(theblas::isamax(3, x.data(), 1) ==
               static_cast<int>(cblas_isamax(3, x.data(), 1) + 1));
        expect_equal(theblas::snrm2(3, x.data(), 1), cblas_snrm2(3, x.data(), 1));
        expect_equal(theblas::sasum(3, x.data(), 1), cblas_sasum(3, x.data(), 1));
    }
    {
        std::array<std::complex<double>, 2> x = {{{1.0, 2.0}, {3.0, -1.0}}};
        std::array<std::complex<double>, 2> y = {{{2.0, 0.0}, {0.0, 1.0}}};
        std::complex<double> dotu1{};
        std::complex<double> dotu2{};
        std::complex<double> dotc1{};
        std::complex<double> dotc2{};
        dotu1 = theblas::zdotu(2, x.data(), 1, y.data(), 1);
        dotc1 = theblas::zdotc(2, x.data(), 1, y.data(), 1);
        cblas_zdotu_sub(2, x.data(), 1, y.data(), 1, &dotu2);
        cblas_zdotc_sub(2, x.data(), 1, y.data(), 1, &dotc2);
        expect_equal(dotu1, dotu2);
        expect_equal(dotc1, dotc2);
        expect_equal(theblas::dznrm2(2, x.data(), 1), cblas_dznrm2(2, x.data(), 1));
        expect_equal(theblas::dzasum(2, x.data(), 1), cblas_dzasum(2, x.data(), 1));
        assert(theblas::izamax(2, x.data(), 1) ==
               static_cast<int>(cblas_izamax(2, x.data(), 1) + 1));
    }
    {
        std::array<float, 2> x1 = {1.0F, 0.0F};
        std::array<float, 2> y1 = {0.0F, 1.0F};
        auto x2 = x1;
        auto y2 = y1;
        theblas::srot(2, x1.data(), 1, y1.data(), 1, 0.0F, 1.0F);
        cblas_srot(2, x2.data(), 1, y2.data(), 1, 0.0F, 1.0F);
        expect_equal(x1, x2);
        expect_equal(y1, y2);
    }
    {
        float a1 = 3.0F;
        float b1 = 4.0F;
        float c1 = 0.0F;
        float s1 = 0.0F;
        float a2 = a1;
        float b2 = b1;
        float c2 = c1;
        float s2 = s1;
        theblas::srotg(&a1, &b1, &c1, &s1);
        cblas_srotg(&a2, &b2, &c2, &s2);
        expect_equal(a1, a2);
        expect_equal(b1, b2);
        expect_equal(c1, c2);
        expect_equal(s1, s2);
    }
    {
        std::array<float, 2> x1 = {1.0F, 2.0F};
        std::array<float, 2> y1 = {3.0F, 4.0F};
        const std::array<float, 5> param = {-1.0F, 1.0F, 2.0F, 3.0F, 4.0F};
        auto x2 = x1;
        auto y2 = y1;
        theblas::srotm(2, x1.data(), 1, y1.data(), 1, param.data());
        cblas_srotm(2, x2.data(), 1, y2.data(), 1, param.data());
        expect_equal(x1, x2);
        expect_equal(y1, y2);
    }
    {
        float d11 = 1.0F;
        float d21 = 2.0F;
        float b11 = 3.0F;
        const float b21 = 4.0F;
        std::array<float, 5> p1 = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
        float d12 = d11;
        float d22 = d21;
        float b12 = b11;
        auto p2 = p1;
        theblas::srotmg(&d11, &d21, &b11, b21, p1.data());
        cblas_srotmg(&d12, &d22, &b12, b21, p2.data());
        expect_equal(d11, d12);
        expect_equal(d21, d22);
        expect_equal(b11, b12);
        expect_equal(p1, p2);
    }

    {
        std::array<float, 6> a1 = {1, 3, 5, 2, 4, 6};
        auto a2 = a1;
        std::array<float, 2> x = {1, 1};
        std::array<float, 3> y1 = {0, 0, 0};
        auto y2 = y1;
        theblas::sgemv('N', 3, 2, 1.0F, a1.data(), 3, x.data(), 1, 0.0F, y1.data(), 1);
        cblas_sgemv(CblasColMajor, CblasNoTrans, 3, 2, 1.0F, a2.data(), 3, x.data(), 1, 0.0F,
                    y2.data(), 1);
        expect_equal(y1, y2);
    }
    {
        std::array<std::complex<float>, 6> ab1 = {
            std::complex<float>(0, 0),  std::complex<float>(1, 1), std::complex<float>(0, 1),
            std::complex<float>(2, -1), std::complex<float>(3, 0), std::complex<float>(0, 0)};
        auto ab2 = ab1;
        std::array<std::complex<float>, 2> x = {std::complex<float>(1, 0),
                                                std::complex<float>(1, 0)};
        std::array<std::complex<float>, 2> y1 = {std::complex<float>(0, 0),
                                                 std::complex<float>(0, 0)};
        auto y2 = y1;
        const std::complex<float> alpha(1, 0);
        const std::complex<float> beta(0, 0);
        theblas::cgbmv('C', 2, 2, 1, 1, alpha, ab1.data(), 3, x.data(), 1, beta, y1.data(), 1);
        cblas_cgbmv(CblasColMajor, CblasConjTrans, 2, 2, 1, 1, &alpha, ab2.data(), 3, x.data(), 1,
                    &beta, y2.data(), 1);
        expect_equal(y1, y2);
    }
    {
        std::array<double, 4> a1 = {2, 1, 0, 3};
        auto a2 = a1;
        std::array<double, 2> x = {1, 2};
        std::array<double, 2> y1 = {0, 0};
        auto y2 = y1;
        theblas::dsymv('L', 2, 1.0, a1.data(), 2, x.data(), 1, 0.0, y1.data(), 1);
        cblas_dsymv(CblasColMajor, CblasLower, 2, 1.0, a2.data(), 2, x.data(), 1, 0.0, y2.data(),
                    1);
        expect_equal(y1, y2);
    }
    {
        std::array<std::complex<float>, 4> a1 = {
            std::complex<float>(2, 0), std::complex<float>(0, 0), std::complex<float>(1, 1),
            std::complex<float>(3, 0)};
        auto a2 = a1;
        std::array<std::complex<float>, 2> x = {std::complex<float>(1, 0),
                                                std::complex<float>(1, 0)};
        std::array<std::complex<float>, 2> y1 = {std::complex<float>(0, 0),
                                                 std::complex<float>(0, 0)};
        auto y2 = y1;
        const std::complex<float> alpha(1, 0);
        const std::complex<float> beta(0, 0);
        theblas::chemv('U', 2, alpha, a1.data(), 2, x.data(), 1, beta, y1.data(), 1);
        cblas_chemv(CblasColMajor, CblasUpper, 2, &alpha, a2.data(), 2, x.data(), 1, &beta,
                    y2.data(), 1);
        expect_equal(y1, y2);
    }
    {
        std::array<double, 4> ab1 = {0, 2, -1, 2};
        auto ab2 = ab1;
        std::array<double, 2> x = {1, 1};
        std::array<double, 2> y1 = {0, 0};
        auto y2 = y1;
        theblas::dsbmv('U', 2, 1, 1.0, ab1.data(), 2, x.data(), 1, 0.0, y1.data(), 1);
        cblas_dsbmv(CblasColMajor, CblasUpper, 2, 1, 1.0, ab2.data(), 2, x.data(), 1, 0.0,
                    y2.data(), 1);
        expect_equal(y1, y2);
    }
    {
        using cf = std::complex<float>;
        std::array<cf, 4> ab1 = {cf(0, 0), cf(2, 0), cf(1, 1), cf(3, 0)};
        auto ab2 = ab1;
        std::array<cf, 2> x = {cf(1, 0), cf(1, 0)};
        std::array<cf, 2> y1 = {cf(0, 0), cf(0, 0)};
        auto y2 = y1;
        const cf alpha(1, 0);
        const cf beta(0, 0);
        theblas::chbmv('U', 2, 1, alpha, ab1.data(), 2, x.data(), 1, beta, y1.data(), 1);
        cblas_chbmv(CblasColMajor, CblasUpper, 2, 1, &alpha, ab2.data(), 2, x.data(), 1, &beta,
                    y2.data(), 1);
        expect_equal(y1, y2);
    }
    {
        std::array<double, 3> ap1 = {2, 1, 3};
        auto ap2 = ap1;
        std::array<double, 2> x = {1, 2};
        std::array<double, 2> y1 = {0, 0};
        auto y2 = y1;
        theblas::dspmv('U', 2, 1.0, ap1.data(), x.data(), 1, 0.0, y1.data(), 1);
        cblas_dspmv(CblasColMajor, CblasUpper, 2, 1.0, ap2.data(), x.data(), 1, 0.0, y2.data(), 1);
        expect_equal(y1, y2);
    }
    {
        using cd = std::complex<double>;
        std::array<cd, 3> ap1 = {cd(2, 0), cd(1, 1), cd(3, 0)};
        auto ap2 = ap1;
        std::array<cd, 2> x = {cd(1, 0), cd(1, 0)};
        std::array<cd, 2> y1 = {cd(0, 0), cd(0, 0)};
        auto y2 = y1;
        const cd alpha(1, 0);
        const cd beta(0, 0);
        theblas::zhpmv('U', 2, alpha, ap1.data(), x.data(), 1, beta, y1.data(), 1);
        cblas_zhpmv(CblasColMajor, CblasUpper, 2, &alpha, ap2.data(), x.data(), 1, &beta, y2.data(),
                    1);
        expect_equal(y1, y2);
    }
    {
        std::array<float, 4> a1 = {2, 0, 3, 4};
        auto a2 = a1;
        std::array<float, 2> x1 = {1, 2};
        auto x2 = x1;
        theblas::strmv('U', 'N', 'N', 2, a1.data(), 2, x1.data(), 1);
        cblas_strmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, a2.data(), 2,
                    x2.data(), 1);
        expect_equal(x1, x2);
    }
    {
        std::array<float, 4> a1 = {2, 0, 3, 4};
        auto a2 = a1;
        std::array<float, 2> x1 = {8, 8};
        auto x2 = x1;
        theblas::strsv('U', 'N', 'N', 2, a1.data(), 2, x1.data(), 1);
        cblas_strsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, a2.data(), 2,
                    x2.data(), 1);
        expect_equal(x1, x2);
    }
    {
        std::array<double, 4> ab1 = {2, 3, 4, 0};
        auto ab2 = ab1;
        std::array<double, 2> x1 = {1, 2};
        auto x2 = x1;
        theblas::dtbmv('L', 'N', 'N', 2, 1, ab1.data(), 2, x1.data(), 1);
        cblas_dtbmv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, 2, 1, ab2.data(), 2,
                    x2.data(), 1);
        expect_equal(x1, x2);
    }
    {
        std::array<double, 4> ab1 = {2, 3, 4, 0};
        auto ab2 = ab1;
        std::array<double, 2> x1 = {2, 11};
        auto x2 = x1;
        theblas::dtbsv('L', 'N', 'N', 2, 1, ab1.data(), 2, x1.data(), 1);
        cblas_dtbsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, 2, 1, ab2.data(), 2,
                    x2.data(), 1);
        expect_equal(x1, x2);
    }
    {
        std::array<double, 3> ap1 = {2, 3, 4};
        auto ap2 = ap1;
        std::array<double, 2> x1 = {1, 2};
        auto x2 = x1;
        theblas::dtpmv('L', 'N', 'N', 2, ap1.data(), x1.data(), 1);
        cblas_dtpmv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, 2, ap2.data(), x2.data(),
                    1);
        expect_equal(x1, x2);
    }
    {
        std::array<double, 3> ap1 = {2, 3, 4};
        auto ap2 = ap1;
        std::array<double, 2> x1 = {2, 11};
        auto x2 = x1;
        theblas::dtpsv('L', 'N', 'N', 2, ap1.data(), x1.data(), 1);
        cblas_dtpsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, 2, ap2.data(), x2.data(),
                    1);
        expect_equal(x1, x2);
    }
    {
        std::array<float, 2> x = {1, 2};
        std::array<float, 2> y = {3, 4};
        std::array<float, 4> a1 = {0, 0, 0, 0};
        auto a2 = a1;
        theblas::sger(2, 2, 1.0F, x.data(), 1, y.data(), 1, a1.data(), 2);
        cblas_sger(CblasColMajor, 2, 2, 1.0F, x.data(), 1, y.data(), 1, a2.data(), 2);
        expect_equal(a1, a2);
    }
    {
        using cf = std::complex<float>;
        std::array<cf, 2> x = {cf(1, 1), cf(2, 0)};
        std::array<cf, 2> y = {cf(1, -1), cf(0, 1)};
        std::array<cf, 4> a1 = {cf(0), cf(0), cf(0), cf(0)};
        auto a2 = a1;
        const cf alpha(1, 0);
        theblas::cgeru(2, 2, alpha, x.data(), 1, y.data(), 1, a1.data(), 2);
        cblas_cgeru(CblasColMajor, 2, 2, &alpha, x.data(), 1, y.data(), 1, a2.data(), 2);
        expect_equal(a1, a2);
        a1 = {cf(0), cf(0), cf(0), cf(0)};
        a2 = a1;
        theblas::cgerc(2, 2, alpha, x.data(), 1, y.data(), 1, a1.data(), 2);
        cblas_cgerc(CblasColMajor, 2, 2, &alpha, x.data(), 1, y.data(), 1, a2.data(), 2);
        expect_equal(a1, a2);
    }
    {
        std::array<float, 2> x = {1, 2};
        std::array<float, 4> a1 = {0, 0, 0, 0};
        auto a2 = a1;
        theblas::ssyr('U', 2, 1.0F, x.data(), 1, a1.data(), 2);
        cblas_ssyr(CblasColMajor, CblasUpper, 2, 1.0F, x.data(), 1, a2.data(), 2);
        expect_equal(a1, a2);
    }
    {
        using cf = std::complex<float>;
        std::array<cf, 2> x = {cf(1, 1), cf(2, 0)};
        std::array<cf, 4> a1 = {cf(0), cf(0), cf(0), cf(0)};
        auto a2 = a1;
        theblas::cher('U', 2, 1.0F, x.data(), 1, a1.data(), 2);
        cblas_cher(CblasColMajor, CblasUpper, 2, 1.0F, x.data(), 1, a2.data(), 2);
        expect_equal(a1, a2);
    }
    {
        std::array<float, 2> x = {1, 0};
        std::array<float, 2> y = {0, 1};
        std::array<float, 4> a1 = {0, 0, 0, 0};
        auto a2 = a1;
        theblas::ssyr2('U', 2, 1.0F, x.data(), 1, y.data(), 1, a1.data(), 2);
        cblas_ssyr2(CblasColMajor, CblasUpper, 2, 1.0F, x.data(), 1, y.data(), 1, a2.data(), 2);
        expect_equal(a1, a2);
    }
    {
        using cf = std::complex<float>;
        std::array<cf, 2> x = {cf(1, 0), cf(0, 0)};
        std::array<cf, 2> y = {cf(0, 0), cf(1, 0)};
        std::array<cf, 4> a1 = {cf(0), cf(0), cf(0), cf(0)};
        auto a2 = a1;
        const cf alpha(1, 0);
        theblas::cher2('U', 2, alpha, x.data(), 1, y.data(), 1, a1.data(), 2);
        cblas_cher2(CblasColMajor, CblasUpper, 2, &alpha, x.data(), 1, y.data(), 1, a2.data(), 2);
        expect_equal(a1, a2);
    }
    {
        std::array<float, 2> x = {1, 2};
        std::array<float, 3> ap1 = {0, 0, 0};
        auto ap2 = ap1;
        theblas::sspr('U', 2, 1.0F, x.data(), 1, ap1.data());
        cblas_sspr(CblasColMajor, CblasUpper, 2, 1.0F, x.data(), 1, ap2.data());
        expect_equal(ap1, ap2);
    }
    {
        using cf = std::complex<float>;
        std::array<cf, 2> x = {cf(1, 1), cf(2, 0)};
        std::array<cf, 3> ap1 = {cf(0), cf(0), cf(0)};
        auto ap2 = ap1;
        theblas::chpr('U', 2, 1.0F, x.data(), 1, ap1.data());
        cblas_chpr(CblasColMajor, CblasUpper, 2, 1.0F, x.data(), 1, ap2.data());
        expect_equal(ap1, ap2);
    }
    {
        std::array<float, 2> x = {1, 0};
        std::array<float, 2> y = {0, 1};
        std::array<float, 3> ap1 = {0, 0, 0};
        auto ap2 = ap1;
        theblas::sspr2('U', 2, 1.0F, x.data(), 1, y.data(), 1, ap1.data());
        cblas_sspr2(CblasColMajor, CblasUpper, 2, 1.0F, x.data(), 1, y.data(), 1, ap2.data());
        expect_equal(ap1, ap2);
    }
    {
        using cf = std::complex<float>;
        std::array<cf, 2> x = {cf(1, 0), cf(0, 0)};
        std::array<cf, 2> y = {cf(0, 0), cf(1, 0)};
        std::array<cf, 3> ap1 = {cf(0), cf(0), cf(0)};
        auto ap2 = ap1;
        const cf alpha(1, 0);
        theblas::chpr2('U', 2, alpha, x.data(), 1, y.data(), 1, ap1.data());
        cblas_chpr2(CblasColMajor, CblasUpper, 2, &alpha, x.data(), 1, y.data(), 1, ap2.data());
        expect_equal(ap1, ap2);
    }

    return 0;
}
