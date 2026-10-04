#include "theblas/theblas.h"

#include "test_cases.hpp"
#include "test_support.hpp"

#include <array>
#include <cassert>
#include <complex>
#include <cstring>
#include <type_traits>

namespace theblas::test {
namespace {

int error_param = 0;
const char *error_routine = nullptr;

void record_error(const char *routine, int param) {
    error_routine = routine;
    error_param = param;
}

template <typename T, typename Gemm> void test_gemm(Gemm gemm) {
    const std::array<T, 4> a = {T(1), T(3), T(2), T(4)};
    const std::array<T, 4> identity = {T(1), T(0), T(0), T(1)};
    std::array<T, 4> c{};
    gemm('N', 'N', 2, 2, 2, T(1), a.data(), 2, identity.data(), 2, T(0), c.data(), 2);
    for (std::size_t i = 0; i < c.size(); ++i)
        assert(almost_equal(c[i], a[i]));

    c.fill(T(7));
    gemm('N', 'N', 2, 2, 2, T(0), nullptr, 2, nullptr, 2, T(1), c.data(), 2);
    for (const T value : c)
        assert(almost_equal(value, T(7)));
    gemm('N', 'N', 2, 2, 2, T(0), nullptr, 2, nullptr, 2, T(0), c.data(), 2);
    for (const T value : c)
        assert(almost_equal(value, T(0)));
    gemm('N', 'N', 2, 2, 0, T(1), nullptr, 2, nullptr, 1, T(0), c.data(), 2);
    for (const T value : c)
        assert(almost_equal(value, T(0)));
    gemm('N', 'N', 0, 2, 1, T(1), nullptr, 1, nullptr, 1, T(0), nullptr, 1);
    gemm('N', 'N', 2, 0, 1, T(1), nullptr, 2, nullptr, 1, T(0), nullptr, 2);

    const std::array<T, 6> rectangular = {T(1), T(2), T(3), T(4), T(5), T(6)};
    std::array<T, 6> transposed_product{};
    gemm('T', 'N', 3, 2, 2, T(1), rectangular.data(), 2, identity.data(), 2, T(0),
         transposed_product.data(), 3);
    const std::array<T, 6> expected_transpose = {T(1), T(3), T(5), T(2), T(4), T(6)};
    for (std::size_t i = 0; i < transposed_product.size(); ++i)
        assert(almost_equal(transposed_product[i], expected_transpose[i]));
}

template <typename T, typename Symm> void test_symm(Symm symm) {
    const std::array<T, 4> a = {T(2), T(0), T(1), T(3)};
    const std::array<T, 4> identity = {T(1), T(0), T(0), T(1)};
    std::array<T, 4> c{};
    symm('L', 'U', 2, 2, T(1), a.data(), 2, identity.data(), 2, T(0), c.data(), 2);
    assert(almost_equal(c[0], T(2)));
    assert(almost_equal(c[1], T(1)));
    assert(almost_equal(c[2], T(1)));
    assert(almost_equal(c[3], T(3)));
}

template <typename T, typename Syrk> void test_syrk(Syrk syrk) {
    const std::array<T, 4> identity = {T(1), T(0), T(0), T(1)};
    std::array<T, 4> c = {T(0), T(9), T(9), T(0)};
    syrk('U', 'N', 2, 2, 1, identity.data(), 2, 0, c.data(), 2);
    assert(almost_equal(c[0], T(1)));
    assert(almost_equal(c[2], T(0)));
    assert(almost_equal(c[3], T(1)));
    assert(almost_equal(c[1], T(9)));

    c.fill(T(4));
    syrk('L', 'N', 2, 0, T(1), nullptr, 2, T(1), c.data(), 2);
    for (const T value : c)
        assert(almost_equal(value, T(4)));
    syrk('L', 'N', 2, 0, T(1), nullptr, 2, T(0), c.data(), 2);
    assert(almost_equal(c[0], T(0)));
    assert(almost_equal(c[1], T(0)));
    assert(almost_equal(c[3], T(0)));
}

template <typename T, typename Syr2k> void test_syr2k(Syr2k syr2k) {
    const std::array<T, 4> identity = {T(1), T(0), T(0), T(1)};
    std::array<T, 4> c{};
    syr2k('L', 'N', 2, 2, T(1), identity.data(), 2, identity.data(), 2, T(0), c.data(), 2);
    assert(almost_equal(c[0], T(2)));
    assert(almost_equal(c[1], T(0)));
    assert(almost_equal(c[2], T(0)));
    assert(almost_equal(c[3], T(2)));
}

template <typename T, typename Trmm> void test_trmm(Trmm trmm) {
    const std::array<T, 4> a = {T(2), T(0), T(1), T(3)};
    std::array<T, 4> b = {T(1), T(0), T(0), T(1)};
    trmm('L', 'U', 'N', 'N', 2, 2, T(1), a.data(), 2, b.data(), 2);
    assert(almost_equal(b[0], T(2)));
    assert(almost_equal(b[1], T(0)));
    assert(almost_equal(b[2], T(1)));
    assert(almost_equal(b[3], T(3)));
    trmm('L', 'U', 'N', 'N', 2, 2, T(0), nullptr, 2, b.data(), 2);
    for (const T value : b)
        assert(almost_equal(value, T(0)));
}

template <typename T, typename Trsm> void test_trsm(Trsm trsm) {
    const std::array<T, 4> a = {T(2), T(0), T(1), T(3)};
    std::array<T, 4> b = {T(1), T(0), T(0), T(1)};
    trsm('L', 'U', 'N', 'N', 2, 2, T(1), a.data(), 2, b.data(), 2);
    assert(almost_equal(b[0], T(0.5)));
    assert(almost_equal(b[1], T(0)));
    assert(almost_equal(b[2], T(-1.0 / 6.0)));
    assert(almost_equal(b[3], T(1.0 / 3.0)));
}

template <typename T, typename Scalar, typename Herk> void test_herk(Herk herk) {
    const std::array<T, 4> identity = {T(1), T(0), T(0), T(1)};
    std::array<T, 4> c = {T(0, 4), T(8, 8), T(8, 8), T(0, -4)};
    herk('L', 'N', 2, 2, Scalar(1), identity.data(), 2, Scalar(0), c.data(), 2);
    assert(almost_equal(c[0], T(1, 0)));
    assert(almost_equal(c[1], T(0, 0)));
    assert(almost_equal(c[2], T(8, 8)));
    assert(almost_equal(c[3], T(1, 0)));
}

template <typename T, typename Scalar, typename Hemm> void test_hemm(Hemm hemm) {
    const std::array<T, 4> a = {T(2, 0), T(0, 0), T(1, 1), T(3, 0)};
    const std::array<T, 4> identity = {T(1, 0), T(0, 0), T(0, 0), T(1, 0)};
    std::array<T, 4> c{};
    hemm('L', 'U', 2, 2, Scalar(1, 0), a.data(), 2, identity.data(), 2, Scalar(0, 0), c.data(), 2);
    assert(almost_equal(c[0], T(2, 0)));
    assert(almost_equal(c[1], T(1, -1)));
    assert(almost_equal(c[2], T(1, 1)));
    assert(almost_equal(c[3], T(3, 0)));
}

template <typename T, typename Scalar, typename Beta, typename Her2k> void test_her2k(Her2k her2k) {
    const std::array<T, 4> identity = {T(1, 0), T(0, 0), T(0, 0), T(1, 0)};
    std::array<T, 4> c{};
    her2k('U', 'N', 2, 2, Scalar(1, 1), identity.data(), 2, identity.data(), 2, Beta(0), c.data(),
          2);
    assert(almost_equal(c[0], T(2, 0)));
    assert(almost_equal(c[1], T(0, 0)));
    assert(almost_equal(c[2], T(0, 0)));
    assert(almost_equal(c[3], T(2, 0)));
}

template <typename T> void test_real_routines() {
    test_gemm<T>([](char ta, char tb, int m, int n, int k, T alpha, const T *a, int lda, const T *b,
                    int ldb, T beta, T *c, int ldc) {
        if constexpr (std::is_same_v<T, float>) {
            sgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        } else {
            dgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
    });
    test_symm<T>([](char side, char uplo, int m, int n, T alpha, const T *a, int lda, const T *b,
                    int ldb, T beta, T *c, int ldc) {
        if constexpr (std::is_same_v<T, float>) {
            ssymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        } else {
            dsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }
    });
    test_syrk<T>([](char uplo, char trans, int n, int k, T alpha, const T *a, int lda, T beta, T *c,
                    int ldc) {
        if constexpr (std::is_same_v<T, float>) {
            ssyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        } else {
            dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }
    });
    test_syr2k<T>([](char uplo, char trans, int n, int k, T alpha, const T *a, int lda, const T *b,
                     int ldb, T beta, T *c, int ldc) {
        if constexpr (std::is_same_v<T, float>) {
            ssyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        } else {
            dsyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
    });
    test_trmm<T>([](char side, char uplo, char trans, char diag, int m, int n, T alpha, const T *a,
                    int lda, T *b, int ldb) {
        if constexpr (std::is_same_v<T, float>) {
            strmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
        } else {
            dtrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
        }
    });
    test_trsm<T>([](char side, char uplo, char trans, char diag, int m, int n, T alpha, const T *a,
                    int lda, T *b, int ldb) {
        if constexpr (std::is_same_v<T, float>) {
            strsm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
        } else {
            dtrsm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
        }
    });
}

template <typename T> void test_complex_routines() {
    using R = typename T::value_type;
    test_gemm<T>([](char ta, char tb, int m, int n, int k, T alpha, const T *a, int lda, const T *b,
                    int ldb, T beta, T *c, int ldc) {
        if constexpr (std::is_same_v<R, float>) {
            cgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        } else {
            zgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
    });
    test_trmm<T>([](char side, char uplo, char trans, char diag, int m, int n, T alpha, const T *a,
                    int lda, T *b, int ldb) {
        if constexpr (std::is_same_v<R, float>) {
            ctrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
        } else {
            ztrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
        }
    });
    test_trsm<T>([](char side, char uplo, char trans, char diag, int m, int n, T alpha, const T *a,
                    int lda, T *b, int ldb) {
        if constexpr (std::is_same_v<R, float>) {
            ctrsm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
        } else {
            ztrsm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
        }
    });
    test_hemm<T, T>([](char side, char uplo, int m, int n, T alpha, const T *a, int lda, const T *b,
                       int ldb, T beta, T *c, int ldc) {
        if constexpr (std::is_same_v<R, float>) {
            chemm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        } else {
            zhemm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
        }
    });
    test_herk<T, R>([](char uplo, char trans, int n, int k, R alpha, const T *a, int lda, R beta,
                       T *c, int ldc) {
        if constexpr (std::is_same_v<R, float>) {
            cherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        } else {
            zherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
        }
    });
    test_her2k<T, T, R>([](char uplo, char trans, int n, int k, T alpha, const T *a, int lda,
                           const T *b, int ldb, R beta, T *c, int ldc) {
        if constexpr (std::is_same_v<R, float>) {
            cher2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        } else {
            zher2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
    });
}

} // namespace

void run_level3_tests() {
    test_real_routines<float>();
    test_real_routines<double>();
    test_complex_routines<std::complex<float>>();
    test_complex_routines<std::complex<double>>();

    auto expect_error = [](int param, const char *routine, auto call) {
        error_param = 0;
        error_routine = nullptr;
        const auto previous_handler = set_error_handler(&record_error);
        call();
        set_error_handler(previous_handler);
        assert(error_param == param);
        assert(error_routine != nullptr && std::strcmp(error_routine, routine) == 0);
    };
    expect_error(3, "SGEMM",
                 [] { sgemm('N', 'N', -1, 1, 1, 1.0F, nullptr, 1, nullptr, 1, 0.0F, nullptr, 1); });
    expect_error(1, "SSYMM",
                 [] { ssymm('X', 'U', 1, 1, 1.0F, nullptr, 1, nullptr, 1, 0.0F, nullptr, 1); });
    expect_error(2, "ZHEMM",
                 [] { zhemm('L', 'X', 1, 1, {}, nullptr, 1, nullptr, 1, {}, nullptr, 1); });
    expect_error(7, "DSYRK", [] { dsyrk('U', 'N', 2, 1, 1.0, nullptr, 1, 0.0, nullptr, 2); });
    expect_error(10, "CHERK", [] { cherk('U', 'N', 2, 1, 1.0F, nullptr, 2, 0.0F, nullptr, 1); });
    expect_error(9, "SSYR2K",
                 [] { ssyr2k('U', 'N', 2, 1, 1.0F, nullptr, 2, nullptr, 1, 0.0F, nullptr, 2); });
    expect_error(12, "ZHER2K",
                 [] { zher2k('U', 'N', 2, 1, {}, nullptr, 2, nullptr, 2, 0.0, nullptr, 1); });
    expect_error(9, "CTRMM", [] { ctrmm('L', 'U', 'N', 'N', 2, 1, {}, nullptr, 1, nullptr, 2); });
    expect_error(11, "ZTRSM", [] { ztrsm('L', 'U', 'N', 'N', 2, 2, {}, nullptr, 2, nullptr, 1); });
}

} // namespace theblas::test
