#include "theblas/theblas.h"

#include <array>
#include <cassert>
#include <cmath>
#include <complex>

namespace {

bool almost_equal(float a, float b, float eps = 1e-6F) {
    return std::fabs(a - b) <= eps;
}

bool almost_equal(double a, double b, double eps = 1e-12) {
    return std::fabs(a - b) <= eps;
}

bool almost_equal(std::complex<float> a, std::complex<float> b, float eps = 1e-6F) {
    return almost_equal(a.real(), b.real(), eps) && almost_equal(a.imag(), b.imag(), eps);
}

bool almost_equal(std::complex<double> a, std::complex<double> b, double eps = 1e-12) {
    return almost_equal(a.real(), b.real(), eps) && almost_equal(a.imag(), b.imag(), eps);
}

} // namespace

namespace {

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void test_level1() {
    {
        std::array<float, 3> x = {1.0F, 2.0F, 3.0F};
        std::array<float, 3> y = {4.0F, 5.0F, 6.0F};
        theblas::sswap(3, x.data(), 1, y.data(), 1);
        assert(almost_equal(x[0], 4.0F));
        assert(almost_equal(x[1], 5.0F));
        assert(almost_equal(x[2], 6.0F));
        assert(almost_equal(y[0], 1.0F));
        assert(almost_equal(y[1], 2.0F));
        assert(almost_equal(y[2], 3.0F));
    }

    {
        const std::array<float, 5> x = {1.0F, 9.0F, 2.0F, 9.0F, 3.0F};
        std::array<float, 3> y = {0.0F, 0.0F, 0.0F};
        theblas::scopy(3, x.data(), 2, y.data(), 1);
        assert(almost_equal(y[0], 1.0F));
        assert(almost_equal(y[1], 2.0F));
        assert(almost_equal(y[2], 3.0F));
    }

    {
        const std::array<float, 3> x = {1.0F, 2.0F, 3.0F};
        std::array<float, 3> y = {10.0F, 20.0F, 30.0F};
        theblas::saxpy(3, 0.5F, x.data(), 1, y.data(), 1);
        assert(almost_equal(y[0], 10.5F));
        assert(almost_equal(y[1], 21.0F));
        assert(almost_equal(y[2], 31.5F));
    }

    {
        std::array<double, 3> x = {1.0, -2.0, 3.0};
        theblas::dscal(3, 2.0, x.data(), 1);
        assert(almost_equal(x[0], 2.0));
        assert(almost_equal(x[1], -4.0));
        assert(almost_equal(x[2], 6.0));
    }

    {
        const std::array<float, 3> x = {1.0F, 2.0F, 3.0F};
        const std::array<float, 3> y = {4.0F, 5.0F, 6.0F};
        assert(almost_equal(theblas::sdot(3, x.data(), 1, y.data(), 1), 32.0F));
    }

    {
        const std::array<double, 2> x = {3.0, 4.0};
        assert(almost_equal(theblas::dnrm2(2, x.data(), 1), 5.0));
    }

    {
        const std::array<float, 3> x = {-1.0F, 2.0F, -3.0F};
        assert(almost_equal(theblas::sasum(3, x.data(), 1), 6.0F));
        assert(theblas::isamax(3, x.data(), 1) == 3);
    }

    {
        const std::array<double, 5> x = {1.0, 100.0, 2.0, 90.0, 3.0};
        std::array<double, 3> y = {0.0, 0.0, 0.0};
        theblas::dcopy(3, x.data(), -2, y.data(), 1);
        assert(almost_equal(y[0], 3.0));
        assert(almost_equal(y[1], 2.0));
        assert(almost_equal(y[2], 1.0));
    }

    {
        const std::array<std::complex<float>, 2> x = {{{1.0F, 1.0F}, {2.0F, -1.0F}}};
        std::array<std::complex<float>, 2> y = {{{3.0F, 0.0F}, {1.0F, 1.0F}}};
        theblas::caxpy(2, {2.0F, 0.0F}, x.data(), 1, y.data(), 1);
        assert(almost_equal(y[0], {5.0F, 2.0F}));
        assert(almost_equal(y[1], {5.0F, -1.0F}));
    }

    {
        const std::array<std::complex<double>, 2> x = {{{1.0, 2.0}, {3.0, -1.0}}};
        const std::array<std::complex<double>, 2> y = {{{2.0, 0.0}, {0.0, 1.0}}};
        assert(almost_equal(theblas::zdotu(2, x.data(), 1, y.data(), 1), {3.0, 7.0}));
        assert(almost_equal(theblas::zdotc(2, x.data(), 1, y.data(), 1), {1.0, -1.0}));
    }

    {
        const std::array<std::complex<float>, 2> x = {{{3.0F, 4.0F}, {0.0F, 0.0F}}};
        assert(almost_equal(theblas::scnrm2(2, x.data(), 1), 5.0F));
        assert(almost_equal(theblas::scasum(2, x.data(), 1), 7.0F));
        assert(theblas::icamax(2, x.data(), 1) == 1);
    }

    {
        std::array<float, 2> x = {1.0F, 0.0F};
        std::array<float, 2> y = {0.0F, 1.0F};
        theblas::srot(2, x.data(), 1, y.data(), 1, 0.0F, 1.0F);
        assert(almost_equal(x[0], 0.0F));
        assert(almost_equal(y[0], -1.0F));
        assert(almost_equal(x[1], 1.0F));
        assert(almost_equal(y[1], 0.0F));
    }

    {
        float a = 3.0F;
        float b = 4.0F;
        float c = 0.0F;
        float s = 0.0F;
        theblas::srotg(&a, &b, &c, &s);
        assert(almost_equal(a, 5.0F));
        assert(almost_equal(c, 0.6F));
        assert(almost_equal(s, 0.8F));
    }

    {
        std::array<float, 2> x = {1.0F, 2.0F};
        std::array<float, 2> y = {3.0F, 4.0F};
        const std::array<float, 5> param = {-1.0F, 1.0F, 2.0F, 3.0F, 4.0F};
        theblas::srotm(2, x.data(), 1, y.data(), 1, param.data());
        assert(almost_equal(x[0], 10.0F));
        assert(almost_equal(y[0], 14.0F));
        assert(almost_equal(x[1], 14.0F));
        assert(almost_equal(y[1], 20.0F));
    }

    // --- no-op edge cases (n=0, incx=0, incy=0) ---

    {
        std::array<float, 1> x = {1.0F};
        std::array<float, 1> y = {2.0F};
        theblas::sswap(0, x.data(), 1, y.data(), 1);
        assert(almost_equal(x[0], 1.0F));
        assert(almost_equal(y[0], 2.0F));
    }

    {
        std::array<float, 1> x = {1.0F};
        std::array<float, 1> y = {2.0F};
        theblas::sswap(1, x.data(), 0, y.data(), 1);
        assert(almost_equal(x[0], 1.0F));
        assert(almost_equal(y[0], 2.0F));
    }

    {
        const std::array<float, 1> x = {9.0F};
        std::array<float, 1> y = {7.0F};
        theblas::scopy(1, x.data(), 0, y.data(), 1);
        assert(almost_equal(y[0], 7.0F));
    }

    {
        const std::array<float, 1> x = {9.0F};
        std::array<float, 1> y = {7.0F};
        theblas::saxpy(0, 1.0F, x.data(), 1, y.data(), 1);
        assert(almost_equal(y[0], 7.0F));
    }

    {
        const std::array<float, 1> x = {9.0F};
        std::array<float, 1> y = {7.0F};
        theblas::saxpy(1, 1.0F, x.data(), 1, y.data(), 0);
        assert(almost_equal(y[0], 7.0F));
    }

    {
        const std::array<float, 2> x = {3.0F, 4.0F};
        assert(almost_equal(theblas::sdot(0, x.data(), 1, x.data(), 1), 0.0F));
        assert(almost_equal(theblas::snrm2(0, x.data(), 1), 0.0F));
        assert(almost_equal(theblas::sasum(0, x.data(), 1), 0.0F));
        assert(almost_equal(theblas::sasum(2, x.data(), 0), 0.0F));
        assert(theblas::isamax(0, x.data(), 1) == 0);
        assert(theblas::isamax(2, x.data(), 0) == 0);
        assert(theblas::isamax(2, x.data(), -1) == 0);
    }

    {
        std::array<std::complex<float>, 1> cx = {{{1.0F, 0.0F}}};
        assert(almost_equal(theblas::scasum(0, cx.data(), 1), 0.0F));
        assert(theblas::icamax(0, cx.data(), 1) == 0);
        assert(theblas::icamax(1, cx.data(), 0) == 0);
    }

    // --- dswap ---

    {
        std::array<double, 2> x = {1.0, 2.0};
        std::array<double, 2> y = {3.0, 4.0};
        theblas::dswap(2, x.data(), 1, y.data(), 1);
        assert(almost_equal(x[0], 3.0));
        assert(almost_equal(x[1], 4.0));
        assert(almost_equal(y[0], 1.0));
        assert(almost_equal(y[1], 2.0));
    }

    // --- cswap ---

    {
        std::array<std::complex<float>, 2> x = {{{1.0F, 2.0F}, {3.0F, 4.0F}}};
        std::array<std::complex<float>, 2> y = {{{5.0F, 6.0F}, {7.0F, 8.0F}}};
        theblas::cswap(2, x.data(), 1, y.data(), 1);
        assert(almost_equal(x[0], {5.0F, 6.0F}));
        assert(almost_equal(x[1], {7.0F, 8.0F}));
        assert(almost_equal(y[0], {1.0F, 2.0F}));
        assert(almost_equal(y[1], {3.0F, 4.0F}));
    }

    // --- zswap ---

    {
        std::array<std::complex<double>, 1> x = {{{1.0, 2.0}}};
        std::array<std::complex<double>, 1> y = {{{3.0, 4.0}}};
        theblas::zswap(1, x.data(), 1, y.data(), 1);
        assert(almost_equal(x[0], {3.0, 4.0}));
        assert(almost_equal(y[0], {1.0, 2.0}));
    }

    // --- sswap with negative stride ---

    {
        // negative incx: elements are accessed in reverse order
        std::array<float, 3> x = {1.0F, 2.0F, 3.0F};
        std::array<float, 3> y = {4.0F, 5.0F, 6.0F};
        theblas::sswap(3, x.data(), -1, y.data(), 1);
        assert(almost_equal(x[0], 6.0F));
        assert(almost_equal(x[1], 5.0F));
        assert(almost_equal(x[2], 4.0F));
        assert(almost_equal(y[0], 3.0F));
        assert(almost_equal(y[1], 2.0F));
        assert(almost_equal(y[2], 1.0F));
    }

    // --- ccopy ---

    {
        const std::array<std::complex<float>, 2> x = {{{1.0F, 2.0F}, {3.0F, 4.0F}}};
        std::array<std::complex<float>, 2> y = {{{0.0F, 0.0F}, {0.0F, 0.0F}}};
        theblas::ccopy(2, x.data(), 1, y.data(), 1);
        assert(almost_equal(y[0], {1.0F, 2.0F}));
        assert(almost_equal(y[1], {3.0F, 4.0F}));
    }

    // --- zcopy ---

    {
        const std::array<std::complex<double>, 2> x = {{{1.0, 2.0}, {3.0, 4.0}}};
        std::array<std::complex<double>, 2> y = {{{0.0, 0.0}, {0.0, 0.0}}};
        theblas::zcopy(2, x.data(), 1, y.data(), 1);
        assert(almost_equal(y[0], {1.0, 2.0}));
        assert(almost_equal(y[1], {3.0, 4.0}));
    }

    // --- daxpy ---

    {
        const std::array<double, 3> x = {1.0, 2.0, 3.0};
        std::array<double, 3> y = {10.0, 20.0, 30.0};
        theblas::daxpy(3, 2.0, x.data(), 1, y.data(), 1);
        assert(almost_equal(y[0], 12.0));
        assert(almost_equal(y[1], 24.0));
        assert(almost_equal(y[2], 36.0));
    }

    // --- zaxpy ---

    {
        const std::array<std::complex<double>, 1> x = {{{1.0, 1.0}}};
        std::array<std::complex<double>, 1> y = {{{2.0, 3.0}}};
        theblas::zaxpy(1, {2.0, 0.0}, x.data(), 1, y.data(), 1);
        assert(almost_equal(y[0], {4.0, 5.0}));
    }

    // --- sscal ---

    {
        std::array<float, 3> x = {1.0F, 2.0F, 3.0F};
        theblas::sscal(3, 3.0F, x.data(), 1);
        assert(almost_equal(x[0], 3.0F));
        assert(almost_equal(x[1], 6.0F));
        assert(almost_equal(x[2], 9.0F));
    }

    // --- cscal ---

    {
        std::array<std::complex<float>, 2> x = {{{1.0F, 0.0F}, {0.0F, 1.0F}}};
        theblas::cscal(2, {0.0F, 1.0F}, x.data(), 1);
        assert(almost_equal(x[0], {0.0F, 1.0F}));
        assert(almost_equal(x[1], {-1.0F, 0.0F}));
    }

    // --- zscal ---

    {
        std::array<std::complex<double>, 1> x = {{{2.0, 1.0}}};
        theblas::zscal(1, {0.0, 1.0}, x.data(), 1);
        assert(almost_equal(x[0], {-1.0, 2.0}));
    }

    // --- csscal ---

    {
        std::array<std::complex<float>, 2> x = {{{2.0F, 4.0F}, {1.0F, -1.0F}}};
        theblas::csscal(2, 0.5F, x.data(), 1);
        assert(almost_equal(x[0], {1.0F, 2.0F}));
        assert(almost_equal(x[1], {0.5F, -0.5F}));
    }

    // --- zdscal ---

    {
        std::array<std::complex<double>, 1> x = {{{3.0, 6.0}}};
        theblas::zdscal(1, 2.0, x.data(), 1);
        assert(almost_equal(x[0], {6.0, 12.0}));
    }

    // --- ddot ---

    {
        const std::array<double, 3> x = {1.0, 2.0, 3.0};
        const std::array<double, 3> y = {4.0, 5.0, 6.0};
        assert(almost_equal(theblas::ddot(3, x.data(), 1, y.data(), 1), 32.0));
    }

    // --- cdotu, cdotc ---

    {
        const std::array<std::complex<float>, 2> x = {{{1.0F, 2.0F}, {3.0F, -1.0F}}};
        const std::array<std::complex<float>, 2> y = {{{2.0F, 0.0F}, {0.0F, 1.0F}}};
        assert(almost_equal(theblas::cdotu(2, x.data(), 1, y.data(), 1), {3.0F, 7.0F}));
        assert(almost_equal(theblas::cdotc(2, x.data(), 1, y.data(), 1), {1.0F, -1.0F}));
    }

    // --- snrm2 ---

    {
        const std::array<float, 2> x = {3.0F, 4.0F};
        assert(almost_equal(theblas::snrm2(2, x.data(), 1), 5.0F));
    }

    // --- dznrm2 ---

    {
        const std::array<std::complex<double>, 1> x = {{{3.0, 4.0}}};
        assert(almost_equal(theblas::dznrm2(1, x.data(), 1), 5.0));
    }

    // --- dasum ---

    {
        const std::array<double, 3> x = {-1.0, 2.0, -3.0};
        assert(almost_equal(theblas::dasum(3, x.data(), 1), 6.0));
    }

    // --- dzasum ---

    {
        const std::array<std::complex<double>, 2> x = {{{3.0, 4.0}, {1.0, -2.0}}};
        assert(almost_equal(theblas::dzasum(2, x.data(), 1), 10.0));
    }

    // --- idamax ---

    {
        const std::array<double, 3> x = {1.0, -5.0, 3.0};
        assert(theblas::idamax(3, x.data(), 1) == 2);
    }

    {
        const std::array<double, 2> x = {1.0, 2.0};
        assert(theblas::idamax(0, x.data(), 1) == 0);
        assert(theblas::idamax(2, x.data(), 0) == 0);
        assert(theblas::idamax(2, x.data(), -1) == 0);
    }

    // --- izamax ---

    {
        const std::array<std::complex<double>, 3> x = {{{1.0, 1.0}, {3.0, 4.0}, {0.0, 0.0}}};
        assert(theblas::izamax(3, x.data(), 1) == 2);
    }

    // --- drot ---

    {
        std::array<double, 2> x = {1.0, 0.0};
        std::array<double, 2> y = {0.0, 1.0};
        theblas::drot(2, x.data(), 1, y.data(), 1, 0.0, 1.0);
        assert(almost_equal(x[0], 0.0));
        assert(almost_equal(y[0], -1.0));
        assert(almost_equal(x[1], 1.0));
        assert(almost_equal(y[1], 0.0));
    }

    // --- csrot ---

    {
        std::array<std::complex<float>, 1> x = {{{1.0F, 0.0F}}};
        std::array<std::complex<float>, 1> y = {{{0.0F, 1.0F}}};
        theblas::csrot(1, x.data(), 1, y.data(), 1, 0.0F, 1.0F);
        assert(almost_equal(x[0], {0.0F, 1.0F}));
        assert(almost_equal(y[0], {-1.0F, 0.0F}));
    }

    // --- zdrot ---

    {
        std::array<std::complex<double>, 1> x = {{{1.0, 0.0}}};
        std::array<std::complex<double>, 1> y = {{{0.0, 1.0}}};
        theblas::zdrot(1, x.data(), 1, y.data(), 1, 0.0, 1.0);
        assert(almost_equal(x[0], {0.0, 1.0}));
        assert(almost_equal(y[0], {-1.0, 0.0}));
    }

    // --- drotg ---

    {
        double a = 3.0;
        double b = 4.0;
        double c = 0.0;
        double s = 0.0;
        theblas::drotg(&a, &b, &c, &s);
        assert(almost_equal(a, 5.0));
        assert(almost_equal(c, 0.6));
        assert(almost_equal(s, 0.8));
    }

    // --- srotg with zero inputs ---

    {
        float a = 0.0F;
        float b = 0.0F;
        float c = 0.0F;
        float s = 0.0F;
        theblas::srotg(&a, &b, &c, &s);
        assert(almost_equal(c, 1.0F));
        assert(almost_equal(s, 0.0F));
        assert(almost_equal(a, 0.0F));
        assert(almost_equal(b, 0.0F));
    }

    // --- crotg with zero a ---

    {
        std::complex<float> a = {0.0F, 0.0F};
        std::complex<float> b = {3.0F, 4.0F};
        float c = 0.0F;
        std::complex<float> s = {0.0F, 0.0F};
        theblas::crotg(&a, b, &c, &s);
        assert(almost_equal(c, 0.0F));
        assert(almost_equal(s, {1.0F, 0.0F}));
        assert(almost_equal(a, {3.0F, 4.0F}));
    }

    // --- zrotg with zero a ---

    {
        std::complex<double> a = {0.0, 0.0};
        std::complex<double> b = {5.0, 0.0};
        double c = 0.0;
        std::complex<double> s = {0.0, 0.0};
        theblas::zrotg(&a, b, &c, &s);
        assert(almost_equal(c, 0.0));
        assert(almost_equal(s, {1.0, 0.0}));
        assert(almost_equal(a, {5.0, 0.0}));
    }

    // --- drotm ---

    {
        std::array<double, 2> x = {1.0, 2.0};
        std::array<double, 2> y = {3.0, 4.0};
        const std::array<double, 5> param = {-1.0, 1.0, 2.0, 3.0, 4.0};
        theblas::drotm(2, x.data(), 1, y.data(), 1, param.data());
        assert(almost_equal(x[0], 10.0));
        assert(almost_equal(y[0], 14.0));
        assert(almost_equal(x[1], 14.0));
        assert(almost_equal(y[1], 20.0));
    }

    // --- srotm flag=-2 (no-op) ---

    {
        std::array<float, 2> x = {1.0F, 2.0F};
        std::array<float, 2> y = {3.0F, 4.0F};
        const std::array<float, 5> param = {-2.0F, 0.0F, 0.0F, 0.0F, 0.0F};
        theblas::srotm(2, x.data(), 1, y.data(), 1, param.data());
        assert(almost_equal(x[0], 1.0F));
        assert(almost_equal(y[0], 3.0F));
        assert(almost_equal(x[1], 2.0F));
        assert(almost_equal(y[1], 4.0F));
    }

    // --- srotm flag=0 (h11=h22=1, use h12 and h21) ---

    {
        std::array<float, 1> x = {1.0F};
        std::array<float, 1> y = {2.0F};
        const std::array<float, 5> param = {0.0F, 0.0F, 3.0F, 2.0F, 0.0F};
        theblas::srotm(1, x.data(), 1, y.data(), 1, param.data());
        assert(almost_equal(x[0], 5.0F));
        assert(almost_equal(y[0], 5.0F));
    }

    // --- srotm flag=1 (h12=h21=-1, use h11 and h22) ---

    {
        std::array<float, 1> x = {2.0F};
        std::array<float, 1> y = {3.0F};
        const std::array<float, 5> param = {1.0F, 2.0F, 0.0F, 0.0F, 4.0F};
        theblas::srotm(1, x.data(), 1, y.data(), 1, param.data());
        assert(almost_equal(x[0], 7.0F));
        assert(almost_equal(y[0], 10.0F));
    }

    // --- srotmg: invalid d1 <= 0 sets param[0]=-2 ---

    {
        float d1 = -1.0F;
        float d2 = 1.0F;
        float b1 = 1.0F;
        const float b2 = 1.0F;
        std::array<float, 5> param = {};
        theblas::srotmg(&d1, &d2, &b1, b2, param.data());
        assert(almost_equal(param[0], -2.0F));
    }

    // --- srotmg: valid inputs with b2=0 ---

    {
        float d1 = 1.0F;
        float d2 = 1.0F;
        float b1 = 1.0F;
        const float b2 = 0.0F;
        std::array<float, 5> param = {};
        theblas::srotmg(&d1, &d2, &b1, b2, param.data());
        assert(almost_equal(param[0], -1.0F));
    }

    // --- drotmg: invalid d1=0 sets param[0]=-2 ---

    {
        double d1 = 0.0;
        double d2 = 1.0;
        double b1 = 1.0;
        const double b2 = 1.0;
        std::array<double, 5> param = {};
        theblas::drotmg(&d1, &d2, &b1, b2, param.data());
        assert(almost_equal(param[0], -2.0));
    }

    // --- drotmg: valid inputs with b2=0 ---

    {
        double d1 = 1.0;
        double d2 = 1.0;
        double b1 = 1.0;
        const double b2 = 0.0;
        std::array<double, 5> param = {};
        theblas::drotmg(&d1, &d2, &b1, b2, param.data());
        assert(almost_equal(param[0], -1.0));
    }

    // --- n=0 / incx=0 no-ops for d/c/z variants not yet covered ---

    {
        std::array<double, 1> x = {1.0};
        std::array<double, 1> y = {2.0};
        theblas::dswap(0, x.data(), 1, y.data(), 1);
        assert(almost_equal(x[0], 1.0));
        assert(almost_equal(y[0], 2.0));
    }

    {
        std::array<std::complex<float>, 1> x = {{{1.0F, 0.0F}}};
        std::array<std::complex<float>, 1> y = {{{2.0F, 0.0F}}};
        theblas::cswap(0, x.data(), 1, y.data(), 1);
        assert(almost_equal(x[0], {1.0F, 0.0F}));
    }

    {
        std::array<std::complex<double>, 1> x = {{{1.0, 0.0}}};
        std::array<std::complex<double>, 1> y = {{{2.0, 0.0}}};
        theblas::zswap(0, x.data(), 1, y.data(), 1);
        assert(almost_equal(x[0], {1.0, 0.0}));
    }

    {
        const std::array<double, 1> x = {9.0};
        std::array<double, 1> y = {7.0};
        theblas::dcopy(0, x.data(), 1, y.data(), 1);
        assert(almost_equal(y[0], 7.0));
    }

    {
        const std::array<std::complex<float>, 1> x = {{{9.0F, 0.0F}}};
        std::array<std::complex<float>, 1> y = {{{7.0F, 0.0F}}};
        theblas::ccopy(0, x.data(), 1, y.data(), 1);
        assert(almost_equal(y[0], {7.0F, 0.0F}));
    }

    {
        const std::array<std::complex<double>, 1> x = {{{9.0, 0.0}}};
        std::array<std::complex<double>, 1> y = {{{7.0, 0.0}}};
        theblas::zcopy(0, x.data(), 1, y.data(), 1);
        assert(almost_equal(y[0], {7.0, 0.0}));
    }

    {
        const std::array<double, 1> x = {1.0};
        std::array<double, 1> y = {10.0};
        theblas::daxpy(0, 2.0, x.data(), 1, y.data(), 1);
        assert(almost_equal(y[0], 10.0));
    }

    {
        const std::array<std::complex<float>, 1> x = {{{1.0F, 0.0F}}};
        std::array<std::complex<float>, 1> y = {{{2.0F, 0.0F}}};
        theblas::caxpy(0, {1.0F, 0.0F}, x.data(), 1, y.data(), 1);
        assert(almost_equal(y[0], {2.0F, 0.0F}));
    }

    {
        const std::array<std::complex<double>, 1> x = {{{1.0, 0.0}}};
        std::array<std::complex<double>, 1> y = {{{2.0, 0.0}}};
        theblas::zaxpy(0, {1.0, 0.0}, x.data(), 1, y.data(), 1);
        assert(almost_equal(y[0], {2.0, 0.0}));
    }

    {
        std::array<double, 1> x = {5.0};
        theblas::dscal(0, 2.0, x.data(), 1);
        assert(almost_equal(x[0], 5.0));
    }

    {
        std::array<std::complex<float>, 1> x = {{{1.0F, 1.0F}}};
        theblas::cscal(0, {2.0F, 0.0F}, x.data(), 1);
        assert(almost_equal(x[0], {1.0F, 1.0F}));
    }

    {
        std::array<std::complex<double>, 1> x = {{{1.0, 1.0}}};
        theblas::zscal(0, {2.0, 0.0}, x.data(), 1);
        assert(almost_equal(x[0], {1.0, 1.0}));
    }

    {
        std::array<std::complex<float>, 1> x = {{{2.0F, 2.0F}}};
        theblas::csscal(0, 2.0F, x.data(), 1);
        assert(almost_equal(x[0], {2.0F, 2.0F}));
    }

    {
        std::array<std::complex<double>, 1> x = {{{3.0, 3.0}}};
        theblas::zdscal(0, 2.0, x.data(), 1);
        assert(almost_equal(x[0], {3.0, 3.0}));
    }

    {
        const std::array<double, 1> x = {1.0};
        const std::array<double, 1> y = {2.0};
        assert(almost_equal(theblas::ddot(0, x.data(), 1, y.data(), 1), 0.0));
    }

    {
        const std::array<std::complex<float>, 1> x = {{{1.0F, 0.0F}}};
        const std::array<std::complex<float>, 1> y = {{{1.0F, 0.0F}}};
        assert(almost_equal(theblas::cdotu(0, x.data(), 1, y.data(), 1), {0.0F, 0.0F}));
        assert(almost_equal(theblas::cdotc(0, x.data(), 1, y.data(), 1), {0.0F, 0.0F}));
    }

    {
        const std::array<std::complex<double>, 1> x = {{{1.0, 0.0}}};
        const std::array<std::complex<double>, 1> y = {{{1.0, 0.0}}};
        assert(almost_equal(theblas::zdotu(0, x.data(), 1, y.data(), 1), {0.0, 0.0}));
        assert(almost_equal(theblas::zdotc(0, x.data(), 1, y.data(), 1), {0.0, 0.0}));
    }

    {
        const std::array<double, 1> x = {1.0};
        assert(almost_equal(theblas::dnrm2(0, x.data(), 1), 0.0));
    }

    {
        const std::array<std::complex<float>, 1> x = {{{1.0F, 0.0F}}};
        assert(almost_equal(theblas::scnrm2(0, x.data(), 1), 0.0F));
    }

    {
        const std::array<std::complex<double>, 1> x = {{{1.0, 0.0}}};
        assert(almost_equal(theblas::dznrm2(0, x.data(), 1), 0.0));
    }

    {
        const std::array<double, 1> x = {1.0};
        assert(almost_equal(theblas::dasum(0, x.data(), 1), 0.0));
    }

    {
        const std::array<std::complex<double>, 1> x = {{{1.0, 0.0}}};
        assert(almost_equal(theblas::dzasum(0, x.data(), 1), 0.0));
    }

    {
        const std::array<std::complex<double>, 1> x = {{{1.0, 0.0}}};
        assert(theblas::izamax(0, x.data(), 1) == 0);
        assert(theblas::izamax(1, x.data(), 0) == 0);
        assert(theblas::izamax(1, x.data(), -1) == 0);
    }

    // --- icamax: update branch (later element is the maximum) ---

    {
        const std::array<std::complex<float>, 2> x = {{{1.0F, 0.0F}, {3.0F, 4.0F}}};
        assert(theblas::icamax(2, x.data(), 1) == 2);
    }

    // --- rot early returns for d/c/z variants ---

    {
        std::array<float, 1> x = {1.0F};
        std::array<float, 1> y = {2.0F};
        theblas::srot(0, x.data(), 1, y.data(), 1, 1.0F, 0.0F);
        assert(almost_equal(x[0], 1.0F));
    }

    {
        std::array<double, 1> x = {1.0};
        std::array<double, 1> y = {2.0};
        theblas::drot(0, x.data(), 1, y.data(), 1, 1.0, 0.0);
        assert(almost_equal(x[0], 1.0));
    }

    {
        std::array<std::complex<float>, 1> x = {{{1.0F, 0.0F}}};
        std::array<std::complex<float>, 1> y = {{{2.0F, 0.0F}}};
        theblas::csrot(0, x.data(), 1, y.data(), 1, 1.0F, 0.0F);
        assert(almost_equal(x[0], {1.0F, 0.0F}));
    }

    {
        std::array<std::complex<double>, 1> x = {{{1.0, 0.0}}};
        std::array<std::complex<double>, 1> y = {{{2.0, 0.0}}};
        theblas::zdrot(0, x.data(), 1, y.data(), 1, 1.0, 0.0);
        assert(almost_equal(x[0], {1.0, 0.0}));
    }

    // --- drotg: zero-scale (a=0, b=0) ---

    {
        double a = 0.0;
        double b = 0.0;
        double c = 0.0;
        double s = 0.0;
        theblas::drotg(&a, &b, &c, &s);
        assert(almost_equal(c, 1.0));
        assert(almost_equal(s, 0.0));
        assert(almost_equal(a, 0.0));
        assert(almost_equal(b, 0.0));
    }

    // --- crotg: non-zero a ---

    {
        // a = 3+4i, b = 0: norm = |a| = 5, alpha = a/|a| = (0.6+0.8i)
        // c = |a|/norm = 1, s = alpha*conj(0)/5 = 0, a_out = alpha*5 = 3+4i
        std::complex<float> a = {3.0F, 4.0F};
        std::complex<float> b = {0.0F, 0.0F};
        float c = 0.0F;
        std::complex<float> s = {0.0F, 0.0F};
        theblas::crotg(&a, b, &c, &s);
        assert(almost_equal(c, 1.0F));
    }

    // --- zrotg: non-zero a ---

    {
        // a = 3+0i, b = 4+0i: norm = 5, alpha = 1+0i
        // c = 3/5 = 0.6, s = 1*conj(4+0i)/5 = 0.8, a = 1*5 = 5
        std::complex<double> a = {3.0, 0.0};
        std::complex<double> b = {4.0, 0.0};
        double c = 0.0;
        std::complex<double> s = {0.0, 0.0};
        theblas::zrotg(&a, b, &c, &s);
        assert(almost_equal(c, 0.6));
        assert(almost_equal(s, {0.8, 0.0}));
        assert(almost_equal(a, {5.0, 0.0}));
    }

    // --- srotm / drotm: n=0 early return ---

    {
        std::array<float, 1> x = {1.0F};
        std::array<float, 1> y = {2.0F};
        const std::array<float, 5> param = {-1.0F, 1.0F, 0.0F, 0.0F, 1.0F};
        theblas::srotm(0, x.data(), 1, y.data(), 1, param.data());
        assert(almost_equal(x[0], 1.0F));
    }

    {
        std::array<double, 1> x = {1.0};
        std::array<double, 1> y = {2.0};
        const std::array<double, 5> param = {-1.0, 1.0, 0.0, 0.0, 1.0};
        theblas::drotm(0, x.data(), 1, y.data(), 1, param.data());
        assert(almost_equal(x[0], 1.0));
    }

    // --- drotm: flag=-2 (no-op) ---

    {
        std::array<double, 2> x = {1.0, 2.0};
        std::array<double, 2> y = {3.0, 4.0};
        const std::array<double, 5> param = {-2.0, 0.0, 0.0, 0.0, 0.0};
        theblas::drotm(2, x.data(), 1, y.data(), 1, param.data());
        assert(almost_equal(x[0], 1.0));
        assert(almost_equal(y[0], 3.0));
    }

    // --- drotm: flag=0 (h11=h22=1 implied, use h12 and h21) ---

    {
        std::array<double, 1> x = {1.0};
        std::array<double, 1> y = {2.0};
        const std::array<double, 5> param = {0.0, 0.0, 3.0, 2.0, 0.0};
        theblas::drotm(1, x.data(), 1, y.data(), 1, param.data());
        assert(almost_equal(x[0], 5.0));
        assert(almost_equal(y[0], 5.0));
    }

    // --- drotm: flag=1 (h12=h21=-1 implied, use h11 and h22) ---

    {
        std::array<double, 1> x = {2.0};
        std::array<double, 1> y = {3.0};
        const std::array<double, 5> param = {1.0, 2.0, 0.0, 0.0, 4.0};
        theblas::drotm(1, x.data(), 1, y.data(), 1, param.data());
        assert(almost_equal(x[0], 7.0));
        assert(almost_equal(y[0], 10.0));
    }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void test_level2_core() {

    /* ================================================================
     * Level 2 BLAS tests
     * ================================================================ */

    /* --- gemv --- */
    // sgemv: y ← 1·a·x + 0·y, a = [[1,2],[3,4],[5,6]], x = [1,1], y = [0,0,0]
    // column-major a: col0={1,3,5}, col1={2,4,6}
    // a·x = [1+2, 3+4, 5+6] = [3, 7, 11]
    {
        std::array<float, 6> a = {1, 3, 5, 2, 4, 6}; // 3x2 col-major
        std::array<float, 2> x = {1, 1};
        std::array<float, 3> y = {0, 0, 0};
        theblas::sgemv('N', 3, 2, 1.0F, a.data(), 3, x.data(), 1, 0.0F, y.data(), 1);
        assert(almost_equal(y[0], 3.0F));
        assert(almost_equal(y[1], 7.0F));
        assert(almost_equal(y[2], 11.0F));
    }
    // dgemv transpose: y ← 1·Aᵀ·x + 0·y
    // Aᵀ = [[1,3,5],[2,4,6]], x = [1,1,1], Aᵀ·x = [9, 12]
    {
        std::array<double, 6> a = {1, 3, 5, 2, 4, 6};
        std::array<double, 3> x = {1, 1, 1};
        std::array<double, 2> y = {0, 0};
        theblas::dgemv('T', 3, 2, 1.0, a.data(), 3, x.data(), 1, 0.0, y.data(), 1);
        assert(almost_equal(y[0], 9.0));
        assert(almost_equal(y[1], 12.0));
    }
    // dgemv with alpha and beta
    {
        std::array<double, 4> a = {1, 0, 0, 1}; // 2x2 identity
        std::array<double, 2> x = {3, 4};
        std::array<double, 2> y = {10, 20};
        theblas::dgemv('N', 2, 2, 2.0, a.data(), 2, x.data(), 1, 3.0, y.data(), 1);
        // y = 2*I*[3,4] + 3*[10,20] = [6,8] + [30,60] = [36, 68]
        assert(almost_equal(y[0], 36.0));
        assert(almost_equal(y[1], 68.0));
    }
    // cgemv conjugate-transpose
    {
        using cf = std::complex<float>;
        // a = [[1+i, 2+0i],[0+0i, 0+i]], col-major: col0={1+i, 0}, col1={2, i}
        std::array<cf, 4> a = {cf(1, 1), cf(0, 0), cf(2, 0), cf(0, 1)};
        std::array<cf, 2> x = {cf(1, 0), cf(1, 0)};
        std::array<cf, 2> y = {cf(0, 0), cf(0, 0)};
        // Aᴴ·x: row0 of Aᴴ = conj(col0 of a) = [1-i, 0], row1 = [2, -i]
        // y[0] = (1-i)*1 + 0*1 = 1-i, y[1] = 2*1 + (-i)*1 = 2-i
        theblas::cgemv('C', 2, 2, cf(1, 0), a.data(), 2, x.data(), 1, cf(0, 0), y.data(), 1);
        assert(almost_equal(y[0], cf(1, -1)));
        assert(almost_equal(y[1], cf(2, -1)));
    }
    // gemv edge case: m=0
    {
        std::array<float, 1> y = {99.0F};
        theblas::sgemv('N', 0, 1, 1.0F, nullptr, 1, nullptr, 1, 0.0F, y.data(), 1);
        assert(almost_equal(y[0], 99.0F)); // no-op
    }

    /* --- symv --- */
    // ssymv upper: a = [[2,1],[1,3]], x = [1,2], y ← 1·a·x + 0·y = [4, 7]
    {
        std::array<float, 4> a = {2, 0, 1, 3}; // col-major, upper stored: (0,0)=2, (0,1)=1, (1,1)=3
        std::array<float, 2> x = {1, 2};
        std::array<float, 2> y = {0, 0};
        theblas::ssymv('U', 2, 1.0F, a.data(), 2, x.data(), 1, 0.0F, y.data(), 1);
        assert(almost_equal(y[0], 4.0F));
        assert(almost_equal(y[1], 7.0F));
    }
    // dsymv lower
    {
        std::array<double, 4> a = {2, 1, 0, 3}; // col-major, lower stored: (0,0)=2, (1,0)=1, (1,1)=3
        std::array<double, 2> x = {1, 2};
        std::array<double, 2> y = {0, 0};
        theblas::dsymv('L', 2, 1.0, a.data(), 2, x.data(), 1, 0.0, y.data(), 1);
        assert(almost_equal(y[0], 4.0));
        assert(almost_equal(y[1], 7.0));
    }

    /* --- hemv --- */
    // chemv: Hermitian a = [[2, 1+i],[1-i, 3]], x=[1,1]
    // y = a·x = [2+(1+i), (1-i)+3] = [3+i, 4-i]
    {
        using cf = std::complex<float>;
        // col-major upper: (0,0)=2, (0,1)=1+i, (1,1)=3
        std::array<cf, 4> a = {cf(2, 0), cf(0, 0), cf(1, 1), cf(3, 0)};
        std::array<cf, 2> x = {cf(1, 0), cf(1, 0)};
        std::array<cf, 2> y = {cf(0, 0), cf(0, 0)};
        theblas::chemv('U', 2, cf(1, 0), a.data(), 2, x.data(), 1, cf(0, 0), y.data(), 1);
        assert(almost_equal(y[0], cf(3, 1)));
        assert(almost_equal(y[1], cf(4, -1)));
    }

    /* --- trmv --- */
    // strmv upper no-trans non-unit: a = [[2,3],[0,4]], x=[1,2] → x ← a·x = [8, 8]
    {
        std::array<float, 4> a = {2, 0, 3, 4}; // col-major
        std::array<float, 2> x = {1, 2};
        theblas::strmv('U', 'N', 'N', 2, a.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], 8.0F));
        assert(almost_equal(x[1], 8.0F));
    }
    // dtrmv lower trans unit-diag: a = [[1,0],[5,1]], Aᵀ = [[1,5],[0,1]], x=[2,3] → [2+15, 3] = [17, 3]
    {
        std::array<double, 4> a = {1, 5, 0, 1}; // col-major lower
        std::array<double, 2> x = {2, 3};
        theblas::dtrmv('L', 'T', 'U', 2, a.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], 17.0));
        assert(almost_equal(x[1], 3.0));
    }

    /* --- trsv --- */
    // strsv upper: a = [[2,3],[0,4]], solve a·x = b with b=[8,8]
    // x[1] = 8/4 = 2, x[0] = (8 - 3*2)/2 = 1
    {
        std::array<float, 4> a = {2, 0, 3, 4};
        std::array<float, 2> x = {8, 8};
        theblas::strsv('U', 'N', 'N', 2, a.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], 1.0F));
        assert(almost_equal(x[1], 2.0F));
    }
    // dtrsv lower: a = [[2,0],[3,4]], solve a·x = b with b=[2,11]
    // x[0] = 2/2 = 1, x[1] = (11 - 3*1)/4 = 2
    {
        std::array<double, 4> a = {2, 3, 0, 4};
        std::array<double, 2> x = {2, 11};
        theblas::dtrsv('L', 'N', 'N', 2, a.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], 1.0));
        assert(almost_equal(x[1], 2.0));
    }

    /* --- ger --- */
    // sger: a ← 1·x·yᵀ + a, x=[1,2], y=[3,4], a=zeros(2,2)
    // Result: a = [[3,4],[6,8]]
    {
        std::array<float, 2> x = {1, 2};
        std::array<float, 2> y = {3, 4};
        std::array<float, 4> a = {0, 0, 0, 0};
        theblas::sger(2, 2, 1.0F, x.data(), 1, y.data(), 1, a.data(), 2);
        assert(almost_equal(a[0], 3.0F)); // (0,0)
        assert(almost_equal(a[1], 6.0F)); // (1,0)
        assert(almost_equal(a[2], 4.0F)); // (0,1)
        assert(almost_equal(a[3], 8.0F)); // (1,1)
    }

    /* --- geru / gerc --- */
    {
        using cf = std::complex<float>;
        std::array<cf, 2> x = {cf(1, 1), cf(2, 0)};
        std::array<cf, 2> y = {cf(1, -1), cf(0, 1)};
        std::array<cf, 4> au = {cf(0), cf(0), cf(0), cf(0)};
        std::array<cf, 4> ac = {cf(0), cf(0), cf(0), cf(0)};
        theblas::cgeru(2, 2, cf(1, 0), x.data(), 1, y.data(), 1, au.data(), 2);
        theblas::cgerc(2, 2, cf(1, 0), x.data(), 1, y.data(), 1, ac.data(), 2);
        // geru: a(i,j) = x[i]*y[j] (no conj)
        // (0,0) = (1+i)*(1-i) = 1+1 = 2
        assert(almost_equal(au[0], cf(2, 0)));
        // gerc: a(i,j) = x[i]*conj(y[j])
        // (0,0) = (1+i)*conj(1-i) = (1+i)*(1+i) = 1+2i-1 = 2i
        assert(almost_equal(ac[0], cf(0, 2)));
    }

    /* --- syr --- */
    // ssyr upper: a ← 1·x·xᵀ + a, x=[1,2], a=zeros
    // a = [[1,2],[2,4]], upper stored
    {
        std::array<float, 2> x = {1, 2};
        std::array<float, 4> a = {0, 0, 0, 0};
        theblas::ssyr('U', 2, 1.0F, x.data(), 1, a.data(), 2);
        assert(almost_equal(a[0], 1.0F)); // (0,0)
        assert(almost_equal(a[2], 2.0F)); // (0,1)
        assert(almost_equal(a[3], 4.0F)); // (1,1)
    }

    /* --- her --- */
    {
        using cf = std::complex<float>;
        std::array<cf, 2> x = {cf(1, 1), cf(2, 0)};
        std::array<cf, 4> a = {cf(0), cf(0), cf(0), cf(0)};
        theblas::cher('U', 2, 1.0F, x.data(), 1, a.data(), 2);
        // a(0,0) = x[0]*conj(x[0]) = (1+i)(1-i) = 2 (real)
        assert(almost_equal(a[0], cf(2, 0)));
        // a(0,1) = x[0]*conj(x[1]) = (1+i)*2 = 2+2i
        assert(almost_equal(a[2], cf(2, 2)));
        // a(1,1) = x[1]*conj(x[1]) = 4 (real)
        assert(almost_equal(a[3], cf(4, 0)));
    }

    /* --- syr2 --- */
    {
        std::array<float, 2> x = {1, 0};
        std::array<float, 2> y = {0, 1};
        std::array<float, 4> a = {0, 0, 0, 0};
        theblas::ssyr2('U', 2, 1.0F, x.data(), 1, y.data(), 1, a.data(), 2);
        // a ← x·yᵀ + y·xᵀ = [[0,1],[1,0]] + [[0,0],[0,0]]... wait
        // x·yᵀ = [[0,1],[0,0]], y·xᵀ = [[0,0],[1,0]]
        // sum = [[0,1],[1,0]], upper: (0,0)=0, (0,1)=1, (1,1)=0
        assert(almost_equal(a[0], 0.0F));
        assert(almost_equal(a[2], 1.0F)); // (0,1)
        assert(almost_equal(a[3], 0.0F));
    }

    /* --- her2 --- */
    {
        using cf = std::complex<float>;
        std::array<cf, 2> x = {cf(1, 0), cf(0, 0)};
        std::array<cf, 2> y = {cf(0, 0), cf(1, 0)};
        std::array<cf, 4> a = {cf(0), cf(0), cf(0), cf(0)};
        theblas::cher2('U', 2, cf(1, 0), x.data(), 1, y.data(), 1, a.data(), 2);
        // a ← α·x·yᴴ + conj(α)·y·xᴴ
        // x·yᴴ = [[0,1],[0,0]], y·xᴴ = [[0,0],[1,0]]
        // upper: (0,1) = 1
        assert(almost_equal(a[2], cf(1, 0))); // (0,1)
    }

    /* --- gbmv --- */
    // sgbmv: 3x3 tridiag a=[2,-1,0; -1,2,-1; 0,-1,2], kl=1, ku=1
    // Band storage (lda=3): col0=[*,-1,2], but standard: row ku+i-j in col j
    // With ku=1,kl=1, lda=1+1+1=3
    // col0: row0=* (ku-0+0=1 → padding), row1=a(0,0)=2, row2=a(1,0)=-1
    // col1: row0=a(0,1)=-1, row1=a(1,1)=2, row2=a(2,1)=-1
    // col2: row0=a(1,2)=-1, row1=a(2,2)=2, row2=* (padding)
    {
        std::array<float, 9> ab = {0, 2, -1,  -1, 2, -1,  -1, 2, 0};
        std::array<float, 3> x = {1, 2, 3};
        std::array<float, 3> y = {0, 0, 0};
        // a·x = [2-2, -1+4-3, -2+6] = [0, 0, 4]
        theblas::sgbmv('N', 3, 3, 1, 1, 1.0F, ab.data(), 3, x.data(), 1, 0.0F, y.data(), 1);
        assert(almost_equal(y[0], 0.0F));
        assert(almost_equal(y[1], 0.0F));
        assert(almost_equal(y[2], 4.0F));
    }

    /* --- sbmv --- */
    // dsbmv upper: symmetric banded, same tridiag [2,-1;-1,2], k=1
    // Upper band storage (lda=2): col0=[*,2], col1=[-1,2]
    {
        std::array<double, 4> ab = {0, 2, -1, 2}; // upper, k=1
        std::array<double, 2> x = {1, 1};
        std::array<double, 2> y = {0, 0};
        // a = [[2,-1],[-1,2]], a·x = [1, 1]
        theblas::dsbmv('U', 2, 1, 1.0, ab.data(), 2, x.data(), 1, 0.0, y.data(), 1);
        assert(almost_equal(y[0], 1.0));
        assert(almost_equal(y[1], 1.0));
    }

    /* --- hbmv --- */
    {
        using cf = std::complex<float>;
        // 2x2 Hermitian with k=0 (diagonal only): a=[[2,0],[0,3]]
        std::array<cf, 2> ab = {cf(2, 0), cf(3, 0)}; // upper, k=0, lda=1
        std::array<cf, 2> x = {cf(1, 0), cf(1, 0)};
        std::array<cf, 2> y = {cf(0, 0), cf(0, 0)};
        theblas::chbmv('U', 2, 0, cf(1, 0), ab.data(), 1, x.data(), 1, cf(0, 0), y.data(), 1);
        assert(almost_equal(y[0], cf(2, 0)));
        assert(almost_equal(y[1], cf(3, 0)));
    }

    /* --- tbmv --- */
    // stbmv: upper triangular banded 2x2, k=1, a=[[2,3],[0,4]]
    // Band upper storage (lda=2): col0=[*,2], col1=[3,4]
    {
        std::array<float, 4> ab = {0, 2, 3, 4};
        std::array<float, 2> x = {1, 2};
        theblas::stbmv('U', 'N', 'N', 2, 1, ab.data(), 2, x.data(), 1);
        // x ← a·x = [2+6, 8] = [8, 8]
        assert(almost_equal(x[0], 8.0F));
        assert(almost_equal(x[1], 8.0F));
    }

    /* --- tbsv --- */
    {
        std::array<float, 4> ab = {0, 2, 3, 4};
        std::array<float, 2> x = {8, 8};
        theblas::stbsv('U', 'N', 'N', 2, 1, ab.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], 1.0F));
        assert(almost_equal(x[1], 2.0F));
    }

    /* --- spmv --- */
    // dspmv upper packed: a = [[2,1],[1,3]], packed = [2,1,3]
    {
        std::array<double, 3> ap = {2, 1, 3};
        std::array<double, 2> x = {1, 2};
        std::array<double, 2> y = {0, 0};
        theblas::dspmv('U', 2, 1.0, ap.data(), x.data(), 1, 0.0, y.data(), 1);
        // a·x = [2+2, 1+6] = [4, 7]
        assert(almost_equal(y[0], 4.0));
        assert(almost_equal(y[1], 7.0));
    }

    /* --- hpmv --- */
    {
        using cf = std::complex<float>;
        // Hermitian a = [[2, 1+i],[1-i, 3]], upper packed = [2, 1+i, 3]
        std::array<cf, 3> ap = {cf(2, 0), cf(1, 1), cf(3, 0)};
        std::array<cf, 2> x = {cf(1, 0), cf(1, 0)};
        std::array<cf, 2> y = {cf(0, 0), cf(0, 0)};
        theblas::chpmv('U', 2, cf(1, 0), ap.data(), x.data(), 1, cf(0, 0), y.data(), 1);
        assert(almost_equal(y[0], cf(3, 1)));
        assert(almost_equal(y[1], cf(4, -1)));
    }

    /* --- tpmv --- */
    // stpmv upper packed: a = [[2,3],[0,4]], packed = [2,3,4]
    {
        std::array<float, 3> ap = {2, 3, 4};
        std::array<float, 2> x = {1, 2};
        theblas::stpmv('U', 'N', 'N', 2, ap.data(), x.data(), 1);
        // x ← a·x = [2+6, 8] = [8, 8]
        assert(almost_equal(x[0], 8.0F));
        assert(almost_equal(x[1], 8.0F));
    }

    /* --- tpsv --- */
    {
        std::array<float, 3> ap = {2, 3, 4};
        std::array<float, 2> x = {8, 8};
        theblas::stpsv('U', 'N', 'N', 2, ap.data(), x.data(), 1);
        assert(almost_equal(x[0], 1.0F));
        assert(almost_equal(x[1], 2.0F));
    }

    /* --- spr --- */
    {
        std::array<float, 2> x = {1, 2};
        std::array<float, 3> ap = {0, 0, 0};
        theblas::sspr('U', 2, 1.0F, x.data(), 1, ap.data());
        // x·xᵀ upper packed: [1, 2, 4]
        assert(almost_equal(ap[0], 1.0F));
        assert(almost_equal(ap[1], 2.0F));
        assert(almost_equal(ap[2], 4.0F));
    }

    /* --- hpr --- */
    {
        using cf = std::complex<float>;
        std::array<cf, 2> x = {cf(1, 1), cf(2, 0)};
        std::array<cf, 3> ap = {cf(0), cf(0), cf(0)};
        theblas::chpr('U', 2, 1.0F, x.data(), 1, ap.data());
        // ap[0] = x[0]*conj(x[0]) = 2 (real)
        assert(almost_equal(ap[0], cf(2, 0)));
        // ap[1] = x[0]*conj(x[1]) = (1+i)*2 = 2+2i
        assert(almost_equal(ap[1], cf(2, 2)));
        // ap[2] = x[1]*conj(x[1]) = 4
        assert(almost_equal(ap[2], cf(4, 0)));
    }

    /* --- spr2 --- */
    {
        std::array<float, 2> x = {1, 0};
        std::array<float, 2> y = {0, 1};
        std::array<float, 3> ap = {0, 0, 0};
        theblas::sspr2('U', 2, 1.0F, x.data(), 1, y.data(), 1, ap.data());
        // x·yᵀ + y·xᵀ upper packed: [0, 1, 0]
        assert(almost_equal(ap[0], 0.0F));
        assert(almost_equal(ap[1], 1.0F));
        assert(almost_equal(ap[2], 0.0F));
    }

    /* --- hpr2 --- */
    {
        using cf = std::complex<float>;
        std::array<cf, 2> x = {cf(1, 0), cf(0, 0)};
        std::array<cf, 2> y = {cf(0, 0), cf(1, 0)};
        std::array<cf, 3> ap = {cf(0), cf(0), cf(0)};
        theblas::chpr2('U', 2, cf(1, 0), x.data(), 1, y.data(), 1, ap.data());
        // (0,1) = α·x[0]·conj(y[1]) + conj(α)·y[0]·conj(x[1]) = 1·1·1 + 1·0·0 = 1
        assert(almost_equal(ap[1], cf(1, 0)));
    }

    /* --- zgemv (double complex) --- */
    {
        using cd = std::complex<double>;
        std::array<cd, 4> a = {cd(1, 0), cd(0, 0), cd(0, 0), cd(1, 0)}; // identity
        std::array<cd, 2> x = {cd(3, 1), cd(4, -2)};
        std::array<cd, 2> y = {cd(0, 0), cd(0, 0)};
        theblas::zgemv('N', 2, 2, cd(1, 0), a.data(), 2, x.data(), 1, cd(0, 0), y.data(), 1);
        assert(almost_equal(y[0], cd(3, 1)));
        assert(almost_equal(y[1], cd(4, -2)));
    }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void test_level2_coverage_variants() {

    /* --- remaining band-matrix precision variants --- */
    {
        // A = [[1, 2], [3, 4]] in general band storage; y = A^T * [1, 1].
        std::array<double, 6> ab = {0, 1, 3, 2, 4, 0};
        std::array<double, 2> x = {1, 1};
        std::array<double, 2> y = {0, 0};
        theblas::dgbmv('T', 2, 2, 1, 1, 1.0, ab.data(), 3, x.data(), 1, 0.0, y.data(), 1);
        assert(almost_equal(y[0], 4.0));
        assert(almost_equal(y[1], 6.0));
    }

    {
        using cf = std::complex<float>;
        // Conjugate-transpose of diagonal(1+i, 2-i).
        std::array<cf, 2> ab = {cf(1, 1), cf(2, -1)};
        std::array<cf, 2> x = {cf(1, 0), cf(1, 0)};
        std::array<cf, 2> y = {cf(0, 0), cf(0, 0)};
        theblas::cgbmv('C', 2, 2, 0, 0, cf(1, 0), ab.data(), 1, x.data(), 1, cf(0, 0),
                       y.data(), 1);
        assert(almost_equal(y[0], cf(1, -1)));
        assert(almost_equal(y[1], cf(2, 1)));
    }

    {
        using cd = std::complex<double>;
        std::array<cd, 2> ab = {cd(2, 0), cd(3, 0)};
        std::array<cd, 2> x = {cd(1, 0), cd(2, 0)};
        std::array<cd, 2> y = {cd(0, 0), cd(0, 0)};
        theblas::zgbmv('N', 2, 2, 0, 0, cd(1, 0), ab.data(), 1, x.data(), 1, cd(0, 0),
                       y.data(), 1);
        assert(almost_equal(y[0], cd(2, 0)));
        assert(almost_equal(y[1], cd(6, 0)));
    }

    {
        // Lower-band storage for [[2, 1], [1, 3]].
        std::array<float, 4> ab = {2, 1, 3, 0};
        std::array<float, 2> x = {1, 2};
        std::array<float, 2> y = {0, 0};
        theblas::ssbmv('L', 2, 1, 1.0F, ab.data(), 2, x.data(), 1, 0.0F, y.data(), 1);
        assert(almost_equal(y[0], 4.0F));
        assert(almost_equal(y[1], 7.0F));
    }

    {
        using cd = std::complex<double>;
        // Lower-band storage for [[2, 1+i], [1-i, 3]].
        std::array<cd, 4> ab = {cd(2, 0), cd(1, -1), cd(3, 0), cd(0, 0)};
        std::array<cd, 2> x = {cd(1, 0), cd(1, 0)};
        std::array<cd, 2> y = {cd(0, 0), cd(0, 0)};
        theblas::zhbmv('L', 2, 1, cd(1, 0), ab.data(), 2, x.data(), 1, cd(0, 0), y.data(),
                       1);
        assert(almost_equal(y[0], cd(3, 1)));
        assert(almost_equal(y[1], cd(4, -1)));
    }

    /* --- remaining triangular precision variants --- */
    {
        using cf = std::complex<float>;
        std::array<cf, 4> a = {cf(2, 1), cf(0, 0), cf(3, -2), cf(4, 1)};
        std::array<cf, 2> x = {cf(1, 0), cf(1, 0)};
        theblas::ctrmv('U', 'C', 'N', 2, a.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], cf(2, -1)));
        assert(almost_equal(x[1], cf(7, 1)));
    }

    {
        using cd = std::complex<double>;
        std::array<cd, 4> a = {cd(0, 0), cd(1, 2), cd(0, 0), cd(0, 0)};
        std::array<cd, 2> x = {cd(1, 0), cd(2, 0)};
        theblas::ztrmv('L', 'N', 'U', 2, a.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], cd(1, 0)));
        assert(almost_equal(x[1], cd(3, 2)));
    }

    {
        using cf = std::complex<float>;
        std::array<cf, 4> a = {cf(2, 1), cf(0, 0), cf(3, -2), cf(4, 1)};
        std::array<cf, 2> x = {cf(2, -1), cf(7, 1)};
        theblas::ctrsv('U', 'C', 'N', 2, a.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], cf(1, 0)));
        assert(almost_equal(x[1], cf(1, 0)));
    }

    {
        using cd = std::complex<double>;
        std::array<cd, 4> a = {cd(0, 0), cd(1, 2), cd(0, 0), cd(0, 0)};
        std::array<cd, 2> x = {cd(1, 0), cd(3, 2)};
        theblas::ztrsv('L', 'N', 'U', 2, a.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], cd(1, 0)));
        assert(almost_equal(x[1], cd(2, 0)));
    }

    {
        std::array<double, 4> ab = {2, 3, 4, 0};
        std::array<double, 2> x = {1, 2};
        theblas::dtbmv('L', 'N', 'N', 2, 1, ab.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], 2.0));
        assert(almost_equal(x[1], 11.0));
    }

    {
        using cf = std::complex<float>;
        std::array<cf, 4> ab = {cf(0, 0), cf(2, 1), cf(3, -2), cf(4, 1)};
        std::array<cf, 2> x = {cf(1, 0), cf(1, 0)};
        theblas::ctbmv('U', 'C', 'N', 2, 1, ab.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], cf(2, -1)));
        assert(almost_equal(x[1], cf(7, 1)));
    }

    {
        using cd = std::complex<double>;
        std::array<cd, 4> ab = {cd(0, 0), cd(1, 2), cd(0, 0), cd(0, 0)};
        std::array<cd, 2> x = {cd(1, 0), cd(1, 0)};
        theblas::ztbmv('L', 'T', 'U', 2, 1, ab.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], cd(2, 2)));
        assert(almost_equal(x[1], cd(1, 0)));
    }

    {
        std::array<double, 4> ab = {2, 3, 4, 0};
        std::array<double, 2> x = {2, 11};
        theblas::dtbsv('L', 'N', 'N', 2, 1, ab.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], 1.0));
        assert(almost_equal(x[1], 2.0));
    }

    {
        using cf = std::complex<float>;
        std::array<cf, 4> ab = {cf(0, 0), cf(2, 1), cf(3, -2), cf(4, 1)};
        std::array<cf, 2> x = {cf(2, -1), cf(7, 1)};
        theblas::ctbsv('U', 'C', 'N', 2, 1, ab.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], cf(1, 0)));
        assert(almost_equal(x[1], cf(1, 0)));
    }

    {
        using cd = std::complex<double>;
        std::array<cd, 4> ab = {cd(0, 0), cd(1, 2), cd(0, 0), cd(0, 0)};
        std::array<cd, 2> x = {cd(2, 2), cd(1, 0)};
        theblas::ztbsv('L', 'T', 'U', 2, 1, ab.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], cd(1, 0)));
        assert(almost_equal(x[1], cd(1, 0)));
    }

    {
        std::array<double, 3> ap = {2, 3, 4};
        std::array<double, 2> x = {1, 2};
        theblas::dtpmv('L', 'N', 'N', 2, ap.data(), x.data(), 1);
        assert(almost_equal(x[0], 2.0));
        assert(almost_equal(x[1], 11.0));
    }

    {
        using cf = std::complex<float>;
        std::array<cf, 3> ap = {cf(2, 1), cf(3, -2), cf(4, 1)};
        std::array<cf, 2> x = {cf(1, 0), cf(1, 0)};
        theblas::ctpmv('U', 'C', 'N', 2, ap.data(), x.data(), 1);
        assert(almost_equal(x[0], cf(2, -1)));
        assert(almost_equal(x[1], cf(7, 1)));
    }

    {
        using cd = std::complex<double>;
        std::array<cd, 3> ap = {cd(0, 0), cd(1, 2), cd(0, 0)};
        std::array<cd, 2> x = {cd(1, 0), cd(1, 0)};
        theblas::ztpmv('L', 'T', 'U', 2, ap.data(), x.data(), 1);
        assert(almost_equal(x[0], cd(2, 2)));
        assert(almost_equal(x[1], cd(1, 0)));
    }

    {
        std::array<double, 3> ap = {2, 3, 4};
        std::array<double, 2> x = {2, 11};
        theblas::dtpsv('L', 'N', 'N', 2, ap.data(), x.data(), 1);
        assert(almost_equal(x[0], 1.0));
        assert(almost_equal(x[1], 2.0));
    }

    {
        using cf = std::complex<float>;
        std::array<cf, 3> ap = {cf(2, 1), cf(3, -2), cf(4, 1)};
        std::array<cf, 2> x = {cf(2, -1), cf(7, 1)};
        theblas::ctpsv('U', 'C', 'N', 2, ap.data(), x.data(), 1);
        assert(almost_equal(x[0], cf(1, 0)));
        assert(almost_equal(x[1], cf(1, 0)));
    }

    {
        using cd = std::complex<double>;
        std::array<cd, 3> ap = {cd(0, 0), cd(1, 2), cd(0, 0)};
        std::array<cd, 2> x = {cd(2, 2), cd(1, 0)};
        theblas::ztpsv('L', 'T', 'U', 2, ap.data(), x.data(), 1);
        assert(almost_equal(x[0], cd(1, 0)));
        assert(almost_equal(x[1], cd(1, 0)));
    }

    /* --- remaining rank-update and symmetric/Hermitian variants --- */
    {
        std::array<double, 2> x = {1, 2};
        std::array<double, 2> y = {3, 4};
        std::array<double, 4> a = {0, 0, 0, 0};
        theblas::dger(2, 2, 1.0, x.data(), 1, y.data(), 1, a.data(), 2);
        assert(almost_equal(a[0], 3.0));
        assert(almost_equal(a[1], 6.0));
        assert(almost_equal(a[2], 4.0));
        assert(almost_equal(a[3], 8.0));
    }

    {
        using cd = std::complex<double>;
        std::array<cd, 2> x = {cd(1, 1), cd(2, 0)};
        std::array<cd, 2> y = {cd(1, -1), cd(0, 1)};
        std::array<cd, 4> unconjugated = {cd(0), cd(0), cd(0), cd(0)};
        std::array<cd, 4> conjugated = {cd(0), cd(0), cd(0), cd(0)};
        theblas::zgeru(2, 2, cd(1, 0), x.data(), 1, y.data(), 1, unconjugated.data(), 2);
        theblas::zgerc(2, 2, cd(1, 0), x.data(), 1, y.data(), 1, conjugated.data(), 2);
        assert(almost_equal(unconjugated[0], cd(2, 0)));
        assert(almost_equal(conjugated[0], cd(0, 2)));
    }

    {
        std::array<double, 2> x = {1, 2};
        std::array<double, 4> a = {0, 0, 0, 0};
        theblas::dsyr('L', 2, 1.0, x.data(), 1, a.data(), 2);
        assert(almost_equal(a[0], 1.0));
        assert(almost_equal(a[1], 2.0));
        assert(almost_equal(a[3], 4.0));
    }

    {
        std::array<double, 2> x = {1, 0};
        std::array<double, 2> y = {0, 1};
        std::array<double, 4> a = {0, 0, 0, 0};
        theblas::dsyr2('L', 2, 1.0, x.data(), 1, y.data(), 1, a.data(), 2);
        assert(almost_equal(a[0], 0.0));
        assert(almost_equal(a[1], 1.0));
        assert(almost_equal(a[3], 0.0));
    }

    {
        using cd = std::complex<double>;
        std::array<cd, 2> x = {cd(1, 1), cd(2, 0)};
        std::array<cd, 4> a = {cd(0), cd(0), cd(0), cd(0)};
        theblas::zher('L', 2, 1.0, x.data(), 1, a.data(), 2);
        assert(almost_equal(a[0], cd(2, 0)));
        assert(almost_equal(a[1], cd(2, -2)));
        assert(almost_equal(a[3], cd(4, 0)));
    }

    {
        using cd = std::complex<double>;
        std::array<cd, 2> x = {cd(1, 0), cd(0, 0)};
        std::array<cd, 2> y = {cd(0, 0), cd(1, 0)};
        std::array<cd, 4> a = {cd(0), cd(0), cd(0), cd(0)};
        theblas::zher2('L', 2, cd(1, 0), x.data(), 1, y.data(), 1, a.data(), 2);
        assert(almost_equal(a[0], cd(0, 0)));
        assert(almost_equal(a[1], cd(1, 0)));
        assert(almost_equal(a[3], cd(0, 0)));
    }

    {
        using cd = std::complex<double>;
        std::array<cd, 4> a = {cd(2, 0), cd(1, -1), cd(0, 0), cd(3, 0)};
        std::array<cd, 2> x = {cd(1, 0), cd(1, 0)};
        std::array<cd, 2> y = {cd(0, 0), cd(0, 0)};
        theblas::zhemv('L', 2, cd(1, 0), a.data(), 2, x.data(), 1, cd(0, 0), y.data(), 1);
        assert(almost_equal(y[0], cd(3, 1)));
        assert(almost_equal(y[1], cd(4, -1)));
    }

    {
        std::array<float, 3> ap = {2, 1, 3};
        std::array<float, 2> x = {1, 2};
        std::array<float, 2> y = {0, 0};
        theblas::sspmv('L', 2, 1.0F, ap.data(), x.data(), 1, 0.0F, y.data(), 1);
        assert(almost_equal(y[0], 4.0F));
        assert(almost_equal(y[1], 7.0F));
    }

    {
        using cd = std::complex<double>;
        std::array<cd, 3> ap = {cd(2, 0), cd(1, -1), cd(3, 0)};
        std::array<cd, 2> x = {cd(1, 0), cd(1, 0)};
        std::array<cd, 2> y = {cd(0, 0), cd(0, 0)};
        theblas::zhpmv('L', 2, cd(1, 0), ap.data(), x.data(), 1, cd(0, 0), y.data(), 1);
        assert(almost_equal(y[0], cd(3, 1)));
        assert(almost_equal(y[1], cd(4, -1)));
    }

    {
        std::array<double, 2> x = {1, 2};
        std::array<double, 3> ap = {0, 0, 0};
        theblas::dspr('L', 2, 1.0, x.data(), 1, ap.data());
        assert(almost_equal(ap[0], 1.0));
        assert(almost_equal(ap[1], 2.0));
        assert(almost_equal(ap[2], 4.0));
    }

    {
        std::array<double, 2> x = {1, 0};
        std::array<double, 2> y = {0, 1};
        std::array<double, 3> ap = {0, 0, 0};
        theblas::dspr2('L', 2, 1.0, x.data(), 1, y.data(), 1, ap.data());
        assert(almost_equal(ap[0], 0.0));
        assert(almost_equal(ap[1], 1.0));
        assert(almost_equal(ap[2], 0.0));
    }

    {
        using cd = std::complex<double>;
        std::array<cd, 2> x = {cd(1, 1), cd(2, 0)};
        std::array<cd, 3> ap = {cd(0), cd(0), cd(0)};
        theblas::zhpr('L', 2, 1.0, x.data(), 1, ap.data());
        assert(almost_equal(ap[0], cd(2, 0)));
        assert(almost_equal(ap[1], cd(2, -2)));
        assert(almost_equal(ap[2], cd(4, 0)));
    }

    {
        using cd = std::complex<double>;
        std::array<cd, 2> x = {cd(1, 0), cd(0, 0)};
        std::array<cd, 2> y = {cd(0, 0), cd(1, 0)};
        std::array<cd, 3> ap = {cd(0), cd(0), cd(0)};
        theblas::zhpr2('L', 2, cd(1, 0), x.data(), 1, y.data(), 1, ap.data());
        assert(almost_equal(ap[0], cd(0, 0)));
        assert(almost_equal(ap[1], cd(1, 0)));
        assert(almost_equal(ap[2], cd(0, 0)));
    }

    /* --- Level 2 no-op and scalar edge cases --- */
    {
        std::array<float, 1> a = {2.0F};
        std::array<float, 1> x = {4.0F};
        std::array<float, 1> y = {3.0F};
        theblas::sgemv('N', 1, 1, 0.0F, a.data(), 1, x.data(), 1, 1.0F, y.data(), 1);
        assert(almost_equal(y[0], 3.0F));
    }

    {
        std::array<float, 1> a = {2.0F};
        std::array<float, 1> x = {4.0F};
        std::array<float, 1> y = {3.0F};
        theblas::ssymv('X', 1, 1.0F, a.data(), 1, x.data(), 1, 1.0F, y.data(), 1);
        assert(almost_equal(y[0], 3.0F));
        theblas::ssymv('U', 1, 0.0F, a.data(), 1, x.data(), 1, 2.0F, y.data(), 1);
        assert(almost_equal(y[0], 6.0F));
    }

    {
        using cf = std::complex<float>;
        std::array<cf, 1> a = {cf(2, 0)};
        std::array<cf, 1> x = {cf(4, 0)};
        std::array<cf, 1> y = {cf(3, 0)};
        theblas::chemv('X', 1, cf(1, 0), a.data(), 1, x.data(), 1, cf(1, 0), y.data(), 1);
        assert(almost_equal(y[0], cf(3, 0)));
        theblas::chemv('U', 1, cf(0, 0), a.data(), 1, x.data(), 1, cf(2, 0), y.data(), 1);
        assert(almost_equal(y[0], cf(6, 0)));
    }

    {
        std::array<float, 1> a = {2.0F};
        std::array<float, 1> x = {4.0F};
        std::array<float, 1> y = {3.0F};
        theblas::sgbmv('X', 1, 1, 0, 0, 1.0F, a.data(), 1, x.data(), 1, 1.0F, y.data(), 1);
        assert(almost_equal(y[0], 3.0F));
        theblas::sgbmv('N', 1, 1, 0, 0, 0.0F, a.data(), 1, x.data(), 1, 2.0F, y.data(), 1);
        assert(almost_equal(y[0], 6.0F));
    }

    {
        std::array<float, 1> a = {2.0F};
        std::array<float, 1> x = {4.0F};
        std::array<float, 1> y = {3.0F};
        theblas::ssbmv('X', 1, 0, 1.0F, a.data(), 1, x.data(), 1, 1.0F, y.data(), 1);
        assert(almost_equal(y[0], 3.0F));
        theblas::ssbmv('U', 1, 0, 0.0F, a.data(), 1, x.data(), 1, 2.0F, y.data(), 1);
        assert(almost_equal(y[0], 6.0F));
    }

    {
        using cf = std::complex<float>;
        std::array<cf, 1> a = {cf(2, 0)};
        std::array<cf, 1> x = {cf(4, 0)};
        std::array<cf, 1> y = {cf(3, 0)};
        theblas::chbmv('X', 1, 0, cf(1, 0), a.data(), 1, x.data(), 1, cf(1, 0), y.data(), 1);
        assert(almost_equal(y[0], cf(3, 0)));
        theblas::chbmv('U', 1, 0, cf(0, 0), a.data(), 1, x.data(), 1, cf(2, 0), y.data(), 1);
        assert(almost_equal(y[0], cf(6, 0)));
    }

    {
        std::array<float, 1> ap = {2.0F};
        std::array<float, 1> x = {4.0F};
        std::array<float, 1> y = {3.0F};
        theblas::sspmv('X', 1, 1.0F, ap.data(), x.data(), 1, 1.0F, y.data(), 1);
        assert(almost_equal(y[0], 3.0F));
        theblas::sspmv('U', 1, 0.0F, ap.data(), x.data(), 1, 2.0F, y.data(), 1);
        assert(almost_equal(y[0], 6.0F));
    }

    {
        using cf = std::complex<float>;
        std::array<cf, 1> ap = {cf(2, 0)};
        std::array<cf, 1> x = {cf(4, 0)};
        std::array<cf, 1> y = {cf(3, 0)};
        theblas::chpmv('X', 1, cf(1, 0), ap.data(), x.data(), 1, cf(1, 0), y.data(), 1);
        assert(almost_equal(y[0], cf(3, 0)));
        theblas::chpmv('U', 1, cf(0, 0), ap.data(), x.data(), 1, cf(2, 0), y.data(), 1);
        assert(almost_equal(y[0], cf(6, 0)));
    }

    {
        std::array<float, 1> x = {2.0F};
        std::array<float, 1> y = {3.0F};
        std::array<float, 1> a = {7.0F};
        theblas::sger(0, 1, 1.0F, x.data(), 1, y.data(), 1, a.data(), 1);
        theblas::sger(1, 1, 0.0F, x.data(), 1, y.data(), 1, a.data(), 1);
        assert(almost_equal(a[0], 7.0F));
    }

    {
        using cf = std::complex<float>;
        std::array<cf, 1> x = {cf(2, 0)};
        std::array<cf, 1> y = {cf(3, 0)};
        std::array<cf, 1> a = {cf(7, 0)};
        theblas::cgeru(0, 1, cf(1, 0), x.data(), 1, y.data(), 1, a.data(), 1);
        theblas::cgeru(1, 1, cf(0, 0), x.data(), 1, y.data(), 1, a.data(), 1);
        theblas::cgerc(0, 1, cf(1, 0), x.data(), 1, y.data(), 1, a.data(), 1);
        theblas::cgerc(1, 1, cf(0, 0), x.data(), 1, y.data(), 1, a.data(), 1);
        assert(almost_equal(a[0], cf(7, 0)));
    }

    {
        std::array<float, 1> x = {2.0F};
        std::array<float, 1> y = {3.0F};
        std::array<float, 1> a = {7.0F};
        theblas::ssyr('U', 0, 1.0F, x.data(), 1, a.data(), 1);
        theblas::ssyr('U', 1, 0.0F, x.data(), 1, a.data(), 1);
        theblas::ssyr2('U', 0, 1.0F, x.data(), 1, y.data(), 1, a.data(), 1);
        theblas::ssyr2('U', 1, 0.0F, x.data(), 1, y.data(), 1, a.data(), 1);
        assert(almost_equal(a[0], 7.0F));
    }

    {
        using cf = std::complex<float>;
        std::array<cf, 1> x = {cf(2, 1)};
        std::array<cf, 1> y = {cf(3, 1)};
        std::array<cf, 1> a = {cf(7, 0)};
        theblas::cher('U', 0, 1.0F, x.data(), 1, a.data(), 1);
        theblas::cher('U', 1, 0.0F, x.data(), 1, a.data(), 1);
        theblas::cher2('U', 0, cf(1, 0), x.data(), 1, y.data(), 1, a.data(), 1);
        theblas::cher2('U', 1, cf(0, 0), x.data(), 1, y.data(), 1, a.data(), 1);
        assert(almost_equal(a[0], cf(7, 0)));
    }

    {
        std::array<float, 1> x = {2.0F};
        std::array<float, 1> y = {3.0F};
        std::array<float, 1> ap = {7.0F};
        theblas::sspr('U', 0, 1.0F, x.data(), 1, ap.data());
        theblas::sspr('U', 1, 0.0F, x.data(), 1, ap.data());
        theblas::sspr2('U', 0, 1.0F, x.data(), 1, y.data(), 1, ap.data());
        theblas::sspr2('U', 1, 0.0F, x.data(), 1, y.data(), 1, ap.data());
        assert(almost_equal(ap[0], 7.0F));
    }

    {
        using cf = std::complex<float>;
        std::array<cf, 1> x = {cf(2, 1)};
        std::array<cf, 1> y = {cf(3, 1)};
        std::array<cf, 1> ap = {cf(7, 0)};
        theblas::chpr('U', 0, 1.0F, x.data(), 1, ap.data());
        theblas::chpr('U', 1, 0.0F, x.data(), 1, ap.data());
        theblas::chpr2('U', 0, cf(1, 0), x.data(), 1, y.data(), 1, ap.data());
        theblas::chpr2('U', 1, cf(0, 0), x.data(), 1, y.data(), 1, ap.data());
        assert(almost_equal(ap[0], cf(7, 0)));
    }

    /* --- Remaining lower-triangular and conjugate paths --- */
    {
        std::array<float, 4> a = {2.0F, 3.0F, 0.0F, 4.0F};
        std::array<float, 2> x = {1.0F, 2.0F};
        theblas::strmv('L', 'N', 'N', 2, a.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], 2.0F));
        assert(almost_equal(x[1], 11.0F));

        x = {1.0F, 1.0F};
        theblas::strmv('L', 'C', 'N', 2, a.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], 5.0F));
        assert(almost_equal(x[1], 4.0F));
    }

    {
        std::array<double, 1> a = {2.0};
        std::array<double, 1> x = {3.0};
        theblas::dtrmv('U', 'C', 'N', 1, a.data(), 1, x.data(), 1);
        assert(almost_equal(x[0], 6.0));
    }

    {
        using cd = std::complex<double>;
        std::array<cd, 1> a = {cd(2, 1)};
        std::array<cd, 1> x = {cd(1, 0)};
        theblas::ztrmv('U', 'C', 'N', 1, a.data(), 1, x.data(), 1);
        assert(almost_equal(x[0], cd(2, -1)));
    }

    {
        std::array<double, 4> a = {2.0, 3.0, 0.0, 4.0};
        std::array<double, 2> x = {5.0, 4.0};
        theblas::dtrsv('L', 'C', 'N', 2, a.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], 1.0));
        assert(almost_equal(x[1], 1.0));
    }

    {
        std::array<double, 4> ab = {2.0, 3.0, 4.0, 0.0};
        std::array<double, 2> x = {1.0, 1.0};
        theblas::dtbmv('L', 'C', 'N', 2, 1, ab.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], 5.0));
        assert(almost_equal(x[1], 4.0));

        x = {5.0, 4.0};
        theblas::dtbsv('L', 'C', 'N', 2, 1, ab.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], 1.0));
        assert(almost_equal(x[1], 1.0));
    }

    {
        // A = [[2, 3], [0, 4]]; solve A^T * x = [2, 7].
        std::array<float, 4> ab = {0.0F, 2.0F, 3.0F, 4.0F};
        std::array<float, 2> x = {2.0F, 7.0F};
        theblas::stbsv('U', 'T', 'N', 2, 1, ab.data(), 2, x.data(), 1);
        assert(almost_equal(x[0], 1.0F));
        assert(almost_equal(x[1], 1.0F));
    }

    {
        std::array<double, 3> ap = {2.0, 3.0, 4.0};
        std::array<double, 2> x = {1.0, 1.0};
        theblas::dtpmv('L', 'C', 'N', 2, ap.data(), x.data(), 1);
        assert(almost_equal(x[0], 5.0));
        assert(almost_equal(x[1], 4.0));

        x = {5.0, 4.0};
        theblas::dtpsv('L', 'C', 'N', 2, ap.data(), x.data(), 1);
        assert(almost_equal(x[0], 1.0));
        assert(almost_equal(x[1], 1.0));
    }

    {
        using cf = std::complex<float>;
        std::array<cf, 4> ab = {cf(0, 0), cf(2, 0), cf(1, 1), cf(3, 0)};
        std::array<cf, 2> x = {cf(1, 0), cf(1, 0)};
        std::array<cf, 2> y = {cf(0, 0), cf(0, 0)};
        theblas::chbmv('U', 2, 1, cf(1, 0), ab.data(), 2, x.data(), 1, cf(0, 0), y.data(), 1);
        assert(almost_equal(y[0], cf(3, 1)));
        assert(almost_equal(y[1], cf(4, -1)));
    }

    {
        std::array<float, 1> a = {2.0F};
        std::array<float, 1> x = {1.0F};
        theblas::strmv('U', 'N', 'N', 0, a.data(), 1, x.data(), 1);
        theblas::strsv('U', 'N', 'N', 0, a.data(), 1, x.data(), 1);
        theblas::stbmv('U', 'N', 'N', 0, 0, a.data(), 1, x.data(), 1);
        theblas::stbsv('U', 'N', 'N', 0, 0, a.data(), 1, x.data(), 1);
        theblas::stpmv('U', 'N', 'N', 0, a.data(), x.data(), 1);
        theblas::stpsv('U', 'N', 'N', 0, a.data(), x.data(), 1);
        assert(almost_equal(x[0], 1.0F));
    }
}

} // namespace

int main() {
    test_level1();
    test_level2_core();
    test_level2_coverage_variants();
    return 0;
}
