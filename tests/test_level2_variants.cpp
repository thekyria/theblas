#include "theblas/theblas.h"

#include "test_support.hpp"

#include <array>
#include <cassert>
#include <complex>

namespace theblas::test {

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void run_level2_coverage_variant_tests() {
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


} // namespace theblas::test
