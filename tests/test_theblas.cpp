#include "theblas/theblas.h"

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

int main() {
    {
        float x[] = {1.0F, 2.0F, 3.0F};
        float y[] = {4.0F, 5.0F, 6.0F};
        theblas::sswap(3, x, 1, y, 1);
        assert(almost_equal(x[0], 4.0F));
        assert(almost_equal(x[1], 5.0F));
        assert(almost_equal(x[2], 6.0F));
        assert(almost_equal(y[0], 1.0F));
        assert(almost_equal(y[1], 2.0F));
        assert(almost_equal(y[2], 3.0F));
    }

    {
        const float x[] = {1.0F, 9.0F, 2.0F, 9.0F, 3.0F};
        float y[] = {0.0F, 0.0F, 0.0F};
        theblas::scopy(3, x, 2, y, 1);
        assert(almost_equal(y[0], 1.0F));
        assert(almost_equal(y[1], 2.0F));
        assert(almost_equal(y[2], 3.0F));
    }

    {
        const float x[] = {1.0F, 2.0F, 3.0F};
        float y[] = {10.0F, 20.0F, 30.0F};
        theblas::saxpy(3, 0.5F, x, 1, y, 1);
        assert(almost_equal(y[0], 10.5F));
        assert(almost_equal(y[1], 21.0F));
        assert(almost_equal(y[2], 31.5F));
    }

    {
        double x[] = {1.0, -2.0, 3.0};
        theblas::dscal(3, 2.0, x, 1);
        assert(almost_equal(x[0], 2.0));
        assert(almost_equal(x[1], -4.0));
        assert(almost_equal(x[2], 6.0));
    }

    {
        const float x[] = {1.0F, 2.0F, 3.0F};
        const float y[] = {4.0F, 5.0F, 6.0F};
        assert(almost_equal(theblas::sdot(3, x, 1, y, 1), 32.0F));
    }

    {
        const double x[] = {3.0, 4.0};
        assert(almost_equal(theblas::dnrm2(2, x, 1), 5.0));
    }

    {
        const float x[] = {-1.0F, 2.0F, -3.0F};
        assert(almost_equal(theblas::sasum(3, x, 1), 6.0F));
        assert(theblas::isamax(3, x, 1) == 3);
    }

    {
        const double x[] = {1.0, 100.0, 2.0, 90.0, 3.0};
        double y[] = {0.0, 0.0, 0.0};
        theblas::dcopy(3, x, -2, y, 1);
        assert(almost_equal(y[0], 3.0));
        assert(almost_equal(y[1], 2.0));
        assert(almost_equal(y[2], 1.0));
    }

    {
        const std::complex<float> x[] = {{1.0F, 1.0F}, {2.0F, -1.0F}};
        std::complex<float> y[] = {{3.0F, 0.0F}, {1.0F, 1.0F}};
        theblas::caxpy(2, {2.0F, 0.0F}, x, 1, y, 1);
        assert(almost_equal(y[0], {5.0F, 2.0F}));
        assert(almost_equal(y[1], {5.0F, -1.0F}));
    }

    {
        const std::complex<double> x[] = {{1.0, 2.0}, {3.0, -1.0}};
        const std::complex<double> y[] = {{2.0, 0.0}, {0.0, 1.0}};
        assert(almost_equal(theblas::zdotu(2, x, 1, y, 1), {3.0, 7.0}));
        assert(almost_equal(theblas::zdotc(2, x, 1, y, 1), {1.0, -1.0}));
    }

    {
        const std::complex<float> x[] = {{3.0F, 4.0F}, {0.0F, 0.0F}};
        assert(almost_equal(theblas::scnrm2(2, x, 1), 5.0F));
        assert(almost_equal(theblas::scasum(2, x, 1), 7.0F));
        assert(theblas::icamax(2, x, 1) == 1);
    }

    {
        float x[] = {1.0F, 0.0F};
        float y[] = {0.0F, 1.0F};
        theblas::srot(2, x, 1, y, 1, 0.0F, 1.0F);
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
        float x[] = {1.0F, 2.0F};
        float y[] = {3.0F, 4.0F};
        const float param[] = {-1.0F, 1.0F, 2.0F, 3.0F, 4.0F};
        theblas::srotm(2, x, 1, y, 1, param);
        assert(almost_equal(x[0], 10.0F));
        assert(almost_equal(y[0], 14.0F));
        assert(almost_equal(x[1], 14.0F));
        assert(almost_equal(y[1], 20.0F));
    }

    // --- no-op edge cases (n=0, incx=0, incy=0) ---

    {
        float x[] = {1.0F};
        float y[] = {2.0F};
        theblas::sswap(0, x, 1, y, 1);
        assert(almost_equal(x[0], 1.0F));
        assert(almost_equal(y[0], 2.0F));
    }

    {
        float x[] = {1.0F};
        float y[] = {2.0F};
        theblas::sswap(1, x, 0, y, 1);
        assert(almost_equal(x[0], 1.0F));
        assert(almost_equal(y[0], 2.0F));
    }

    {
        const float x[] = {9.0F};
        float y[] = {7.0F};
        theblas::scopy(1, x, 0, y, 1);
        assert(almost_equal(y[0], 7.0F));
    }

    {
        const float x[] = {9.0F};
        float y[] = {7.0F};
        theblas::saxpy(0, 1.0F, x, 1, y, 1);
        assert(almost_equal(y[0], 7.0F));
    }

    {
        const float x[] = {9.0F};
        float y[] = {7.0F};
        theblas::saxpy(1, 1.0F, x, 1, y, 0);
        assert(almost_equal(y[0], 7.0F));
    }

    {
        const float x[] = {3.0F, 4.0F};
        assert(almost_equal(theblas::sdot(0, x, 1, x, 1), 0.0F));
        assert(almost_equal(theblas::snrm2(0, x, 1), 0.0F));
        assert(almost_equal(theblas::sasum(0, x, 1), 0.0F));
        assert(almost_equal(theblas::sasum(2, x, 0), 0.0F));
        assert(theblas::isamax(0, x, 1) == 0);
        assert(theblas::isamax(2, x, 0) == 0);
        assert(theblas::isamax(2, x, -1) == 0);
    }

    {
        std::complex<float> cx[] = {{1.0F, 0.0F}};
        assert(almost_equal(theblas::scasum(0, cx, 1), 0.0F));
        assert(theblas::icamax(0, cx, 1) == 0);
        assert(theblas::icamax(1, cx, 0) == 0);
    }

    // --- dswap ---

    {
        double x[] = {1.0, 2.0};
        double y[] = {3.0, 4.0};
        theblas::dswap(2, x, 1, y, 1);
        assert(almost_equal(x[0], 3.0));
        assert(almost_equal(x[1], 4.0));
        assert(almost_equal(y[0], 1.0));
        assert(almost_equal(y[1], 2.0));
    }

    // --- cswap ---

    {
        std::complex<float> x[] = {{1.0F, 2.0F}, {3.0F, 4.0F}};
        std::complex<float> y[] = {{5.0F, 6.0F}, {7.0F, 8.0F}};
        theblas::cswap(2, x, 1, y, 1);
        assert(almost_equal(x[0], {5.0F, 6.0F}));
        assert(almost_equal(x[1], {7.0F, 8.0F}));
        assert(almost_equal(y[0], {1.0F, 2.0F}));
        assert(almost_equal(y[1], {3.0F, 4.0F}));
    }

    // --- zswap ---

    {
        std::complex<double> x[] = {{1.0, 2.0}};
        std::complex<double> y[] = {{3.0, 4.0}};
        theblas::zswap(1, x, 1, y, 1);
        assert(almost_equal(x[0], {3.0, 4.0}));
        assert(almost_equal(y[0], {1.0, 2.0}));
    }

    // --- sswap with negative stride ---

    {
        // incx=-1: start_index(3,-1)=2, so addresses x[2],x[1],x[0]
        float x[] = {1.0F, 2.0F, 3.0F};
        float y[] = {4.0F, 5.0F, 6.0F};
        theblas::sswap(3, x, -1, y, 1);
        assert(almost_equal(x[0], 6.0F));
        assert(almost_equal(x[1], 5.0F));
        assert(almost_equal(x[2], 4.0F));
        assert(almost_equal(y[0], 3.0F));
        assert(almost_equal(y[1], 2.0F));
        assert(almost_equal(y[2], 1.0F));
    }

    // --- ccopy ---

    {
        const std::complex<float> x[] = {{1.0F, 2.0F}, {3.0F, 4.0F}};
        std::complex<float> y[] = {{0.0F, 0.0F}, {0.0F, 0.0F}};
        theblas::ccopy(2, x, 1, y, 1);
        assert(almost_equal(y[0], {1.0F, 2.0F}));
        assert(almost_equal(y[1], {3.0F, 4.0F}));
    }

    // --- zcopy ---

    {
        const std::complex<double> x[] = {{1.0, 2.0}, {3.0, 4.0}};
        std::complex<double> y[] = {{0.0, 0.0}, {0.0, 0.0}};
        theblas::zcopy(2, x, 1, y, 1);
        assert(almost_equal(y[0], {1.0, 2.0}));
        assert(almost_equal(y[1], {3.0, 4.0}));
    }

    // --- daxpy ---

    {
        const double x[] = {1.0, 2.0, 3.0};
        double y[] = {10.0, 20.0, 30.0};
        theblas::daxpy(3, 2.0, x, 1, y, 1);
        assert(almost_equal(y[0], 12.0));
        assert(almost_equal(y[1], 24.0));
        assert(almost_equal(y[2], 36.0));
    }

    // --- zaxpy ---

    {
        const std::complex<double> x[] = {{1.0, 1.0}};
        std::complex<double> y[] = {{2.0, 3.0}};
        theblas::zaxpy(1, {2.0, 0.0}, x, 1, y, 1);
        assert(almost_equal(y[0], {4.0, 5.0}));
    }

    // --- sscal ---

    {
        float x[] = {1.0F, 2.0F, 3.0F};
        theblas::sscal(3, 3.0F, x, 1);
        assert(almost_equal(x[0], 3.0F));
        assert(almost_equal(x[1], 6.0F));
        assert(almost_equal(x[2], 9.0F));
    }

    // --- cscal ---

    {
        std::complex<float> x[] = {{1.0F, 0.0F}, {0.0F, 1.0F}};
        theblas::cscal(2, {0.0F, 1.0F}, x, 1);
        // (1+0i)*(0+i) = i; (0+i)*(0+i) = -1
        assert(almost_equal(x[0], {0.0F, 1.0F}));
        assert(almost_equal(x[1], {-1.0F, 0.0F}));
    }

    // --- zscal ---

    {
        std::complex<double> x[] = {{2.0, 1.0}};
        theblas::zscal(1, {0.0, 1.0}, x, 1);
        // (2+i)*(0+i) = 2i + i^2 = -1 + 2i
        assert(almost_equal(x[0], {-1.0, 2.0}));
    }

    // --- csscal ---

    {
        std::complex<float> x[] = {{2.0F, 4.0F}, {1.0F, -1.0F}};
        theblas::csscal(2, 0.5F, x, 1);
        assert(almost_equal(x[0], {1.0F, 2.0F}));
        assert(almost_equal(x[1], {0.5F, -0.5F}));
    }

    // --- zdscal ---

    {
        std::complex<double> x[] = {{3.0, 6.0}};
        theblas::zdscal(1, 2.0, x, 1);
        assert(almost_equal(x[0], {6.0, 12.0}));
    }

    // --- ddot ---

    {
        const double x[] = {1.0, 2.0, 3.0};
        const double y[] = {4.0, 5.0, 6.0};
        assert(almost_equal(theblas::ddot(3, x, 1, y, 1), 32.0));
    }

    // --- cdotu, cdotc ---

    {
        // x = [(1+2i), (3-i)], y = [(2+0i), (0+i)]
        // cdotu: (1+2i)*2 + (3-i)*i = (2+4i) + (3i+1) = {3, 7}
        // cdotc: conj(x) dot y = (1-2i)*2 + (3+i)*i = (2-4i) + (-1+3i) = {1, -1}
        const std::complex<float> x[] = {{1.0F, 2.0F}, {3.0F, -1.0F}};
        const std::complex<float> y[] = {{2.0F, 0.0F}, {0.0F, 1.0F}};
        assert(almost_equal(theblas::cdotu(2, x, 1, y, 1), {3.0F, 7.0F}));
        assert(almost_equal(theblas::cdotc(2, x, 1, y, 1), {1.0F, -1.0F}));
    }

    // --- snrm2 ---

    {
        const float x[] = {3.0F, 4.0F};
        assert(almost_equal(theblas::snrm2(2, x, 1), 5.0F));
    }

    // --- dznrm2 ---

    {
        const std::complex<double> x[] = {{3.0, 4.0}};
        assert(almost_equal(theblas::dznrm2(1, x, 1), 5.0));
    }

    // --- dasum ---

    {
        const double x[] = {-1.0, 2.0, -3.0};
        assert(almost_equal(theblas::dasum(3, x, 1), 6.0));
    }

    // --- dzasum ---

    {
        // |3|+|4| + |1|+|2| = 3+4+1+2 = 10
        const std::complex<double> x[] = {{3.0, 4.0}, {1.0, -2.0}};
        assert(almost_equal(theblas::dzasum(2, x, 1), 10.0));
    }

    // --- idamax ---

    {
        const double x[] = {1.0, -5.0, 3.0};
        assert(theblas::idamax(3, x, 1) == 2);
    }

    {
        const double x[] = {1.0, 2.0};
        assert(theblas::idamax(0, x, 1) == 0);
        assert(theblas::idamax(2, x, 0) == 0);
        assert(theblas::idamax(2, x, -1) == 0);
    }

    // --- izamax ---

    {
        // abs1: 2, 7, 0 -> max at index 1 (0-based) -> returns 2 (1-based)
        const std::complex<double> x[] = {{1.0, 1.0}, {3.0, 4.0}, {0.0, 0.0}};
        assert(theblas::izamax(3, x, 1) == 2);
    }

    // --- drot ---

    {
        double x[] = {1.0, 0.0};
        double y[] = {0.0, 1.0};
        theblas::drot(2, x, 1, y, 1, 0.0, 1.0);
        assert(almost_equal(x[0], 0.0));
        assert(almost_equal(y[0], -1.0));
        assert(almost_equal(x[1], 1.0));
        assert(almost_equal(y[1], 0.0));
    }

    // --- csrot ---

    {
        std::complex<float> x[] = {{1.0F, 0.0F}};
        std::complex<float> y[] = {{0.0F, 1.0F}};
        // c=0, s=1: x = 0*x + 1*y = 0+i; y = 0*y - 1*x = -1+0i
        theblas::csrot(1, x, 1, y, 1, 0.0F, 1.0F);
        assert(almost_equal(x[0], {0.0F, 1.0F}));
        assert(almost_equal(y[0], {-1.0F, 0.0F}));
    }

    // --- zdrot ---

    {
        std::complex<double> x[] = {{1.0, 0.0}};
        std::complex<double> y[] = {{0.0, 1.0}};
        theblas::zdrot(1, x, 1, y, 1, 0.0, 1.0);
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
        double x[] = {1.0, 2.0};
        double y[] = {3.0, 4.0};
        const double param[] = {-1.0, 1.0, 2.0, 3.0, 4.0};
        theblas::drotm(2, x, 1, y, 1, param);
        assert(almost_equal(x[0], 10.0));
        assert(almost_equal(y[0], 14.0));
        assert(almost_equal(x[1], 14.0));
        assert(almost_equal(y[1], 20.0));
    }

    // --- srotm flag=-2 (no-op) ---

    {
        float x[] = {1.0F, 2.0F};
        float y[] = {3.0F, 4.0F};
        const float param[] = {-2.0F, 0.0F, 0.0F, 0.0F, 0.0F};
        theblas::srotm(2, x, 1, y, 1, param);
        assert(almost_equal(x[0], 1.0F));
        assert(almost_equal(y[0], 3.0F));
        assert(almost_equal(x[1], 2.0F));
        assert(almost_equal(y[1], 4.0F));
    }

    // --- srotm flag=0 (h11=h22=1, use h12 and h21) ---

    {
        // flag=0: x = w + h12*z; y = h21*w + z
        // p = {0, h11_ignored, h21=3, h12=2, h22_ignored}
        float x[] = {1.0F};
        float y[] = {2.0F};
        const float param[] = {0.0F, 0.0F, 3.0F, 2.0F, 0.0F};
        theblas::srotm(1, x, 1, y, 1, param);
        assert(almost_equal(x[0], 5.0F));  // 1 + 2*2 = 5
        assert(almost_equal(y[0], 5.0F));  // 3*1 + 2 = 5
    }

    // --- srotm flag=1 (h12=h21=-1, use h11 and h22) ---

    {
        // flag=1: x = h11*w + z; y = -w + h22*z
        // p = {1, h11=2, h21_ignored, h12_ignored, h22=4}
        float x[] = {2.0F};
        float y[] = {3.0F};
        const float param[] = {1.0F, 2.0F, 0.0F, 0.0F, 4.0F};
        theblas::srotm(1, x, 1, y, 1, param);
        assert(almost_equal(x[0], 7.0F));   // 2*2 + 3 = 7
        assert(almost_equal(y[0], 10.0F));  // -2 + 4*3 = 10
    }

    // --- srotmg: invalid d1 <= 0 sets param[0]=-2 ---

    {
        float d1 = -1.0F;
        float d2 = 1.0F;
        float b1 = 1.0F;
        const float b2 = 1.0F;
        float param[5] = {};
        theblas::srotmg(&d1, &d2, &b1, b2, param);
        assert(almost_equal(param[0], -2.0F));
    }

    // --- srotmg: valid inputs with b2=0 ---

    {
        float d1 = 1.0F;
        float d2 = 1.0F;
        float b1 = 1.0F;
        const float b2 = 0.0F;
        float param[5] = {};
        theblas::srotmg(&d1, &d2, &b1, b2, param);
        assert(almost_equal(param[0], -1.0F));
    }

    // --- drotmg: invalid d1=0 sets param[0]=-2 ---

    {
        double d1 = 0.0;
        double d2 = 1.0;
        double b1 = 1.0;
        const double b2 = 1.0;
        double param[5] = {};
        theblas::drotmg(&d1, &d2, &b1, b2, param);
        assert(almost_equal(param[0], -2.0));
    }

    // --- drotmg: valid inputs with b2=0 ---

    {
        double d1 = 1.0;
        double d2 = 1.0;
        double b1 = 1.0;
        const double b2 = 0.0;
        double param[5] = {};
        theblas::drotmg(&d1, &d2, &b1, b2, param);
        assert(almost_equal(param[0], -1.0));
    }

    // --- n=0 / incx=0 no-ops for d/c/z variants not yet covered ---

    {
        double x[] = {1.0};
        double y[] = {2.0};
        theblas::dswap(0, x, 1, y, 1);
        assert(almost_equal(x[0], 1.0));
        assert(almost_equal(y[0], 2.0));
    }

    {
        std::complex<float> x[] = {{1.0F, 0.0F}};
        std::complex<float> y[] = {{2.0F, 0.0F}};
        theblas::cswap(0, x, 1, y, 1);
        assert(almost_equal(x[0], {1.0F, 0.0F}));
    }

    {
        std::complex<double> x[] = {{1.0, 0.0}};
        std::complex<double> y[] = {{2.0, 0.0}};
        theblas::zswap(0, x, 1, y, 1);
        assert(almost_equal(x[0], {1.0, 0.0}));
    }

    {
        const double x[] = {9.0};
        double y[] = {7.0};
        theblas::dcopy(0, x, 1, y, 1);
        assert(almost_equal(y[0], 7.0));
    }

    {
        const std::complex<float> x[] = {{9.0F, 0.0F}};
        std::complex<float> y[] = {{7.0F, 0.0F}};
        theblas::ccopy(0, x, 1, y, 1);
        assert(almost_equal(y[0], {7.0F, 0.0F}));
    }

    {
        const std::complex<double> x[] = {{9.0, 0.0}};
        std::complex<double> y[] = {{7.0, 0.0}};
        theblas::zcopy(0, x, 1, y, 1);
        assert(almost_equal(y[0], {7.0, 0.0}));
    }

    {
        const double x[] = {1.0};
        double y[] = {10.0};
        theblas::daxpy(0, 2.0, x, 1, y, 1);
        assert(almost_equal(y[0], 10.0));
    }

    {
        const std::complex<float> x[] = {{1.0F, 0.0F}};
        std::complex<float> y[] = {{2.0F, 0.0F}};
        theblas::caxpy(0, {1.0F, 0.0F}, x, 1, y, 1);
        assert(almost_equal(y[0], {2.0F, 0.0F}));
    }

    {
        const std::complex<double> x[] = {{1.0, 0.0}};
        std::complex<double> y[] = {{2.0, 0.0}};
        theblas::zaxpy(0, {1.0, 0.0}, x, 1, y, 1);
        assert(almost_equal(y[0], {2.0, 0.0}));
    }

    {
        double x[] = {5.0};
        theblas::dscal(0, 2.0, x, 1);
        assert(almost_equal(x[0], 5.0));
    }

    {
        std::complex<float> x[] = {{1.0F, 1.0F}};
        theblas::cscal(0, {2.0F, 0.0F}, x, 1);
        assert(almost_equal(x[0], {1.0F, 1.0F}));
    }

    {
        std::complex<double> x[] = {{1.0, 1.0}};
        theblas::zscal(0, {2.0, 0.0}, x, 1);
        assert(almost_equal(x[0], {1.0, 1.0}));
    }

    {
        std::complex<float> x[] = {{2.0F, 2.0F}};
        theblas::csscal(0, 2.0F, x, 1);
        assert(almost_equal(x[0], {2.0F, 2.0F}));
    }

    {
        std::complex<double> x[] = {{3.0, 3.0}};
        theblas::zdscal(0, 2.0, x, 1);
        assert(almost_equal(x[0], {3.0, 3.0}));
    }

    {
        const double x[] = {1.0};
        const double y[] = {2.0};
        assert(almost_equal(theblas::ddot(0, x, 1, y, 1), 0.0));
    }

    {
        const std::complex<float> x[] = {{1.0F, 0.0F}};
        const std::complex<float> y[] = {{1.0F, 0.0F}};
        assert(almost_equal(theblas::cdotu(0, x, 1, y, 1), {0.0F, 0.0F}));
        assert(almost_equal(theblas::cdotc(0, x, 1, y, 1), {0.0F, 0.0F}));
    }

    {
        const std::complex<double> x[] = {{1.0, 0.0}};
        const std::complex<double> y[] = {{1.0, 0.0}};
        assert(almost_equal(theblas::zdotu(0, x, 1, y, 1), {0.0, 0.0}));
        assert(almost_equal(theblas::zdotc(0, x, 1, y, 1), {0.0, 0.0}));
    }

    {
        const double x[] = {1.0};
        assert(almost_equal(theblas::dnrm2(0, x, 1), 0.0));
    }

    {
        const std::complex<float> x[] = {{1.0F, 0.0F}};
        assert(almost_equal(theblas::scnrm2(0, x, 1), 0.0F));
    }

    {
        const std::complex<double> x[] = {{1.0, 0.0}};
        assert(almost_equal(theblas::dznrm2(0, x, 1), 0.0));
    }

    {
        const double x[] = {1.0};
        assert(almost_equal(theblas::dasum(0, x, 1), 0.0));
    }

    {
        const std::complex<double> x[] = {{1.0, 0.0}};
        assert(almost_equal(theblas::dzasum(0, x, 1), 0.0));
    }

    {
        const std::complex<double> x[] = {{1.0, 0.0}};
        assert(theblas::izamax(0, x, 1) == 0);
        assert(theblas::izamax(1, x, 0) == 0);
        assert(theblas::izamax(1, x, -1) == 0);
    }

    // --- icamax: update branch (later element is the maximum) ---

    {
        // abs1: element[0]=1, element[1]=7 -> max at index 1 (0-based) -> returns 2 (1-based)
        const std::complex<float> x[] = {{1.0F, 0.0F}, {3.0F, 4.0F}};
        assert(theblas::icamax(2, x, 1) == 2);
    }

    // --- rot early returns for d/c/z variants ---

    {
        float x[] = {1.0F};
        float y[] = {2.0F};
        theblas::srot(0, x, 1, y, 1, 1.0F, 0.0F);
        assert(almost_equal(x[0], 1.0F));
    }

    {
        double x[] = {1.0};
        double y[] = {2.0};
        theblas::drot(0, x, 1, y, 1, 1.0, 0.0);
        assert(almost_equal(x[0], 1.0));
    }

    {
        std::complex<float> x[] = {{1.0F, 0.0F}};
        std::complex<float> y[] = {{2.0F, 0.0F}};
        theblas::csrot(0, x, 1, y, 1, 1.0F, 0.0F);
        assert(almost_equal(x[0], {1.0F, 0.0F}));
    }

    {
        std::complex<double> x[] = {{1.0, 0.0}};
        std::complex<double> y[] = {{2.0, 0.0}};
        theblas::zdrot(0, x, 1, y, 1, 1.0, 0.0);
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
        float x[] = {1.0F};
        float y[] = {2.0F};
        const float param[] = {-1.0F, 1.0F, 0.0F, 0.0F, 1.0F};
        theblas::srotm(0, x, 1, y, 1, param);
        assert(almost_equal(x[0], 1.0F));
    }

    {
        double x[] = {1.0};
        double y[] = {2.0};
        const double param[] = {-1.0, 1.0, 0.0, 0.0, 1.0};
        theblas::drotm(0, x, 1, y, 1, param);
        assert(almost_equal(x[0], 1.0));
    }

    // --- drotm: flag=-2 (no-op) ---

    {
        double x[] = {1.0, 2.0};
        double y[] = {3.0, 4.0};
        const double param[] = {-2.0, 0.0, 0.0, 0.0, 0.0};
        theblas::drotm(2, x, 1, y, 1, param);
        assert(almost_equal(x[0], 1.0));
        assert(almost_equal(y[0], 3.0));
    }

    // --- drotm: flag=0 (h11=h22=1 implied, use h12 and h21) ---

    {
        // p = {0, _, h21=3, h12=2, _}: x = w + h12*z; y = h21*w + z
        double x[] = {1.0};
        double y[] = {2.0};
        const double param[] = {0.0, 0.0, 3.0, 2.0, 0.0};
        theblas::drotm(1, x, 1, y, 1, param);
        assert(almost_equal(x[0], 5.0));
        assert(almost_equal(y[0], 5.0));
    }

    // --- drotm: flag=1 (h12=h21=-1 implied, use h11 and h22) ---

    {
        // p = {1, h11=2, _, _, h22=4}: x = h11*w + z; y = -w + h22*z
        double x[] = {2.0};
        double y[] = {3.0};
        const double param[] = {1.0, 2.0, 0.0, 0.0, 4.0};
        theblas::drotm(1, x, 1, y, 1, param);
        assert(almost_equal(x[0], 7.0));
        assert(almost_equal(y[0], 10.0));
    }

    return 0;
}
