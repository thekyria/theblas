#include "theblas/theblas.h"

#include "test_support.hpp"

#include <array>
#include <cassert>
#include <complex>

namespace theblas::test {

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void run_level2_core_tests() {
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


} // namespace theblas::test
