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
        const float dot = theblas::sdot(3, x, 1, y, 1);
        assert(almost_equal(dot, 32.0F));
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

    return 0;
}
