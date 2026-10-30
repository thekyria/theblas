#include "theblas/theblas_c.h"

#include "theblas/theblas.h"

#include <complex>

namespace {

// std::complex<T> is layout-compatible with T[2] ([complex.numbers]), so interleaved
// arrays can be viewed as complex arrays.
template <typename T> std::complex<T> *cplx(T *p) {
    return reinterpret_cast<std::complex<T> *>(p);
}

template <typename T> const std::complex<T> *cplx(const T *p) {
    return reinterpret_cast<const std::complex<T> *>(p);
}

template <typename T> std::complex<T> scalar(const T *p) {
    return {p[0], p[1]};
}

template <typename T> void store(T *out, std::complex<T> value) {
    out[0] = value.real();
    out[1] = value.imag();
}

} // namespace

extern "C" {

theblas_error_handler_t theblas_set_error_handler(theblas_error_handler_t handler) {
    return theblas::set_error_handler(handler);
}

// Level-1 vector operations
void theblas_sswap(int n, float *x, int incx, float *y, int incy) {
    theblas::sswap(n, x, incx, y, incy);
}
void theblas_dswap(int n, double *x, int incx, double *y, int incy) {
    theblas::dswap(n, x, incx, y, incy);
}
void theblas_cswap(int n, float *x, int incx, float *y, int incy) {
    theblas::cswap(n, cplx(x), incx, cplx(y), incy);
}
void theblas_zswap(int n, double *x, int incx, double *y, int incy) {
    theblas::zswap(n, cplx(x), incx, cplx(y), incy);
}
void theblas_scopy(int n, const float *x, int incx, float *y, int incy) {
    theblas::scopy(n, x, incx, y, incy);
}
void theblas_dcopy(int n, const double *x, int incx, double *y, int incy) {
    theblas::dcopy(n, x, incx, y, incy);
}
void theblas_ccopy(int n, const float *x, int incx, float *y, int incy) {
    theblas::ccopy(n, cplx(x), incx, cplx(y), incy);
}
void theblas_zcopy(int n, const double *x, int incx, double *y, int incy) {
    theblas::zcopy(n, cplx(x), incx, cplx(y), incy);
}
void theblas_saxpy(int n, float alpha, const float *x, int incx, float *y, int incy) {
    theblas::saxpy(n, alpha, x, incx, y, incy);
}
void theblas_daxpy(int n, double alpha, const double *x, int incx, double *y, int incy) {
    theblas::daxpy(n, alpha, x, incx, y, incy);
}
void theblas_caxpy(int n, const float *alpha, const float *x, int incx, float *y, int incy) {
    theblas::caxpy(n, scalar(alpha), cplx(x), incx, cplx(y), incy);
}
void theblas_zaxpy(int n, const double *alpha, const double *x, int incx, double *y, int incy) {
    theblas::zaxpy(n, scalar(alpha), cplx(x), incx, cplx(y), incy);
}
void theblas_sscal(int n, float alpha, float *x, int incx) {
    theblas::sscal(n, alpha, x, incx);
}
void theblas_dscal(int n, double alpha, double *x, int incx) {
    theblas::dscal(n, alpha, x, incx);
}
void theblas_cscal(int n, const float *alpha, float *x, int incx) {
    theblas::cscal(n, scalar(alpha), cplx(x), incx);
}
void theblas_zscal(int n, const double *alpha, double *x, int incx) {
    theblas::zscal(n, scalar(alpha), cplx(x), incx);
}
void theblas_csscal(int n, float alpha, float *x, int incx) {
    theblas::csscal(n, alpha, cplx(x), incx);
}
void theblas_zdscal(int n, double alpha, double *x, int incx) {
    theblas::zdscal(n, alpha, cplx(x), incx);
}
float theblas_sdot(int n, const float *x, int incx, const float *y, int incy) {
    return theblas::sdot(n, x, incx, y, incy);
}
double theblas_ddot(int n, const double *x, int incx, const double *y, int incy) {
    return theblas::ddot(n, x, incx, y, incy);
}
void theblas_cdotu(int n, const float *x, int incx, const float *y, int incy, float *result) {
    store(result, theblas::cdotu(n, cplx(x), incx, cplx(y), incy));
}
void theblas_cdotc(int n, const float *x, int incx, const float *y, int incy, float *result) {
    store(result, theblas::cdotc(n, cplx(x), incx, cplx(y), incy));
}
void theblas_zdotu(int n, const double *x, int incx, const double *y, int incy, double *result) {
    store(result, theblas::zdotu(n, cplx(x), incx, cplx(y), incy));
}
void theblas_zdotc(int n, const double *x, int incx, const double *y, int incy, double *result) {
    store(result, theblas::zdotc(n, cplx(x), incx, cplx(y), incy));
}
float theblas_snrm2(int n, const float *x, int incx) {
    return theblas::snrm2(n, x, incx);
}
double theblas_dnrm2(int n, const double *x, int incx) {
    return theblas::dnrm2(n, x, incx);
}
float theblas_scnrm2(int n, const float *x, int incx) {
    return theblas::scnrm2(n, cplx(x), incx);
}
double theblas_dznrm2(int n, const double *x, int incx) {
    return theblas::dznrm2(n, cplx(x), incx);
}
float theblas_sasum(int n, const float *x, int incx) {
    return theblas::sasum(n, x, incx);
}
double theblas_dasum(int n, const double *x, int incx) {
    return theblas::dasum(n, x, incx);
}
float theblas_scasum(int n, const float *x, int incx) {
    return theblas::scasum(n, cplx(x), incx);
}
double theblas_dzasum(int n, const double *x, int incx) {
    return theblas::dzasum(n, cplx(x), incx);
}
void theblas_srot(int n, float *x, int incx, float *y, int incy, float c, float s) {
    theblas::srot(n, x, incx, y, incy, c, s);
}
void theblas_drot(int n, double *x, int incx, double *y, int incy, double c, double s) {
    theblas::drot(n, x, incx, y, incy, c, s);
}
void theblas_csrot(int n, float *x, int incx, float *y, int incy, float c, float s) {
    theblas::csrot(n, cplx(x), incx, cplx(y), incy, c, s);
}
void theblas_zdrot(int n, double *x, int incx, double *y, int incy, double c, double s) {
    theblas::zdrot(n, cplx(x), incx, cplx(y), incy, c, s);
}
void theblas_srotg(float *a, float *b, float *c, float *s) {
    theblas::srotg(a, b, c, s);
}
void theblas_drotg(double *a, double *b, double *c, double *s) {
    theblas::drotg(a, b, c, s);
}
void theblas_crotg(float *a, const float *b, float *c, float *s) {
    theblas::crotg(cplx(a), scalar(b), c, cplx(s));
}
void theblas_zrotg(double *a, const double *b, double *c, double *s) {
    theblas::zrotg(cplx(a), scalar(b), c, cplx(s));
}
void theblas_srotm(int n, float *x, int incx, float *y, int incy, const float *param) {
    theblas::srotm(n, x, incx, y, incy, param);
}
void theblas_drotm(int n, double *x, int incx, double *y, int incy, const double *param) {
    theblas::drotm(n, x, incx, y, incy, param);
}
void theblas_srotmg(float *d1, float *d2, float *b1, float b2, float *param) {
    theblas::srotmg(d1, d2, b1, b2, param);
}
void theblas_drotmg(double *d1, double *d2, double *b1, double b2, double *param) {
    theblas::drotmg(d1, d2, b1, b2, param);
}
int theblas_isamax(int n, const float *x, int incx) {
    return theblas::isamax(n, x, incx);
}
int theblas_idamax(int n, const double *x, int incx) {
    return theblas::idamax(n, x, incx);
}
int theblas_icamax(int n, const float *x, int incx) {
    return theblas::icamax(n, cplx(x), incx);
}
int theblas_izamax(int n, const double *x, int incx) {
    return theblas::izamax(n, cplx(x), incx);
}

// Level-2 matrix-vector operations
void theblas_sgemv(char trans, int m, int n, float alpha, const float *a, int lda, const float *x,
                   int incx, float beta, float *y, int incy) {
    theblas::sgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
void theblas_dgemv(char trans, int m, int n, double alpha, const double *a, int lda,
                   const double *x, int incx, double beta, double *y, int incy) {
    theblas::dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
void theblas_cgemv(char trans, int m, int n, const float *alpha, const float *a, int lda,
                   const float *x, int incx, const float *beta, float *y, int incy) {
    theblas::cgemv(trans, m, n, scalar(alpha), cplx(a), lda, cplx(x), incx, scalar(beta), cplx(y),
                   incy);
}
void theblas_zgemv(char trans, int m, int n, const double *alpha, const double *a, int lda,
                   const double *x, int incx, const double *beta, double *y, int incy) {
    theblas::zgemv(trans, m, n, scalar(alpha), cplx(a), lda, cplx(x), incx, scalar(beta), cplx(y),
                   incy);
}
void theblas_ssymv(char uplo, int n, float alpha, const float *a, int lda, const float *x, int incx,
                   float beta, float *y, int incy) {
    theblas::ssymv(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}
void theblas_dsymv(char uplo, int n, double alpha, const double *a, int lda, const double *x,
                   int incx, double beta, double *y, int incy) {
    theblas::dsymv(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}
void theblas_chemv(char uplo, int n, const float *alpha, const float *a, int lda, const float *x,
                   int incx, const float *beta, float *y, int incy) {
    theblas::chemv(uplo, n, scalar(alpha), cplx(a), lda, cplx(x), incx, scalar(beta), cplx(y),
                   incy);
}
void theblas_zhemv(char uplo, int n, const double *alpha, const double *a, int lda, const double *x,
                   int incx, const double *beta, double *y, int incy) {
    theblas::zhemv(uplo, n, scalar(alpha), cplx(a), lda, cplx(x), incx, scalar(beta), cplx(y),
                   incy);
}
void theblas_strmv(char uplo, char trans, char diag, int n, const float *a, int lda, float *x,
                   int incx) {
    theblas::strmv(uplo, trans, diag, n, a, lda, x, incx);
}
void theblas_dtrmv(char uplo, char trans, char diag, int n, const double *a, int lda, double *x,
                   int incx) {
    theblas::dtrmv(uplo, trans, diag, n, a, lda, x, incx);
}
void theblas_ctrmv(char uplo, char trans, char diag, int n, const float *a, int lda, float *x,
                   int incx) {
    theblas::ctrmv(uplo, trans, diag, n, cplx(a), lda, cplx(x), incx);
}
void theblas_ztrmv(char uplo, char trans, char diag, int n, const double *a, int lda, double *x,
                   int incx) {
    theblas::ztrmv(uplo, trans, diag, n, cplx(a), lda, cplx(x), incx);
}
void theblas_strsv(char uplo, char trans, char diag, int n, const float *a, int lda, float *x,
                   int incx) {
    theblas::strsv(uplo, trans, diag, n, a, lda, x, incx);
}
void theblas_dtrsv(char uplo, char trans, char diag, int n, const double *a, int lda, double *x,
                   int incx) {
    theblas::dtrsv(uplo, trans, diag, n, a, lda, x, incx);
}
void theblas_ctrsv(char uplo, char trans, char diag, int n, const float *a, int lda, float *x,
                   int incx) {
    theblas::ctrsv(uplo, trans, diag, n, cplx(a), lda, cplx(x), incx);
}
void theblas_ztrsv(char uplo, char trans, char diag, int n, const double *a, int lda, double *x,
                   int incx) {
    theblas::ztrsv(uplo, trans, diag, n, cplx(a), lda, cplx(x), incx);
}
void theblas_sger(int m, int n, float alpha, const float *x, int incx, const float *y, int incy,
                  float *a, int lda) {
    theblas::sger(m, n, alpha, x, incx, y, incy, a, lda);
}
void theblas_dger(int m, int n, double alpha, const double *x, int incx, const double *y, int incy,
                  double *a, int lda) {
    theblas::dger(m, n, alpha, x, incx, y, incy, a, lda);
}
void theblas_cgeru(int m, int n, const float *alpha, const float *x, int incx, const float *y,
                   int incy, float *a, int lda) {
    theblas::cgeru(m, n, scalar(alpha), cplx(x), incx, cplx(y), incy, cplx(a), lda);
}
void theblas_zgeru(int m, int n, const double *alpha, const double *x, int incx, const double *y,
                   int incy, double *a, int lda) {
    theblas::zgeru(m, n, scalar(alpha), cplx(x), incx, cplx(y), incy, cplx(a), lda);
}
void theblas_cgerc(int m, int n, const float *alpha, const float *x, int incx, const float *y,
                   int incy, float *a, int lda) {
    theblas::cgerc(m, n, scalar(alpha), cplx(x), incx, cplx(y), incy, cplx(a), lda);
}
void theblas_zgerc(int m, int n, const double *alpha, const double *x, int incx, const double *y,
                   int incy, double *a, int lda) {
    theblas::zgerc(m, n, scalar(alpha), cplx(x), incx, cplx(y), incy, cplx(a), lda);
}
void theblas_ssyr(char uplo, int n, float alpha, const float *x, int incx, float *a, int lda) {
    theblas::ssyr(uplo, n, alpha, x, incx, a, lda);
}
void theblas_dsyr(char uplo, int n, double alpha, const double *x, int incx, double *a, int lda) {
    theblas::dsyr(uplo, n, alpha, x, incx, a, lda);
}
void theblas_cher(char uplo, int n, float alpha, const float *x, int incx, float *a, int lda) {
    theblas::cher(uplo, n, alpha, cplx(x), incx, cplx(a), lda);
}
void theblas_zher(char uplo, int n, double alpha, const double *x, int incx, double *a, int lda) {
    theblas::zher(uplo, n, alpha, cplx(x), incx, cplx(a), lda);
}
void theblas_ssyr2(char uplo, int n, float alpha, const float *x, int incx, const float *y,
                   int incy, float *a, int lda) {
    theblas::ssyr2(uplo, n, alpha, x, incx, y, incy, a, lda);
}
void theblas_dsyr2(char uplo, int n, double alpha, const double *x, int incx, const double *y,
                   int incy, double *a, int lda) {
    theblas::dsyr2(uplo, n, alpha, x, incx, y, incy, a, lda);
}
void theblas_cher2(char uplo, int n, const float *alpha, const float *x, int incx, const float *y,
                   int incy, float *a, int lda) {
    theblas::cher2(uplo, n, scalar(alpha), cplx(x), incx, cplx(y), incy, cplx(a), lda);
}
void theblas_zher2(char uplo, int n, const double *alpha, const double *x, int incx,
                   const double *y, int incy, double *a, int lda) {
    theblas::zher2(uplo, n, scalar(alpha), cplx(x), incx, cplx(y), incy, cplx(a), lda);
}
void theblas_sgbmv(char trans, int m, int n, int kl, int ku, float alpha, const float *a, int lda,
                   const float *x, int incx, float beta, float *y, int incy) {
    theblas::sgbmv(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}
void theblas_dgbmv(char trans, int m, int n, int kl, int ku, double alpha, const double *a, int lda,
                   const double *x, int incx, double beta, double *y, int incy) {
    theblas::dgbmv(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}
void theblas_cgbmv(char trans, int m, int n, int kl, int ku, const float *alpha, const float *a,
                   int lda, const float *x, int incx, const float *beta, float *y, int incy) {
    theblas::cgbmv(trans, m, n, kl, ku, scalar(alpha), cplx(a), lda, cplx(x), incx, scalar(beta),
                   cplx(y), incy);
}
void theblas_zgbmv(char trans, int m, int n, int kl, int ku, const double *alpha, const double *a,
                   int lda, const double *x, int incx, const double *beta, double *y, int incy) {
    theblas::zgbmv(trans, m, n, kl, ku, scalar(alpha), cplx(a), lda, cplx(x), incx, scalar(beta),
                   cplx(y), incy);
}
void theblas_ssbmv(char uplo, int n, int k, float alpha, const float *a, int lda, const float *x,
                   int incx, float beta, float *y, int incy) {
    theblas::ssbmv(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
void theblas_dsbmv(char uplo, int n, int k, double alpha, const double *a, int lda, const double *x,
                   int incx, double beta, double *y, int incy) {
    theblas::dsbmv(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
void theblas_chbmv(char uplo, int n, int k, const float *alpha, const float *a, int lda,
                   const float *x, int incx, const float *beta, float *y, int incy) {
    theblas::chbmv(uplo, n, k, scalar(alpha), cplx(a), lda, cplx(x), incx, scalar(beta), cplx(y),
                   incy);
}
void theblas_zhbmv(char uplo, int n, int k, const double *alpha, const double *a, int lda,
                   const double *x, int incx, const double *beta, double *y, int incy) {
    theblas::zhbmv(uplo, n, k, scalar(alpha), cplx(a), lda, cplx(x), incx, scalar(beta), cplx(y),
                   incy);
}
void theblas_stbmv(char uplo, char trans, char diag, int n, int k, const float *a, int lda,
                   float *x, int incx) {
    theblas::stbmv(uplo, trans, diag, n, k, a, lda, x, incx);
}
void theblas_dtbmv(char uplo, char trans, char diag, int n, int k, const double *a, int lda,
                   double *x, int incx) {
    theblas::dtbmv(uplo, trans, diag, n, k, a, lda, x, incx);
}
void theblas_ctbmv(char uplo, char trans, char diag, int n, int k, const float *a, int lda,
                   float *x, int incx) {
    theblas::ctbmv(uplo, trans, diag, n, k, cplx(a), lda, cplx(x), incx);
}
void theblas_ztbmv(char uplo, char trans, char diag, int n, int k, const double *a, int lda,
                   double *x, int incx) {
    theblas::ztbmv(uplo, trans, diag, n, k, cplx(a), lda, cplx(x), incx);
}
void theblas_stbsv(char uplo, char trans, char diag, int n, int k, const float *a, int lda,
                   float *x, int incx) {
    theblas::stbsv(uplo, trans, diag, n, k, a, lda, x, incx);
}
void theblas_dtbsv(char uplo, char trans, char diag, int n, int k, const double *a, int lda,
                   double *x, int incx) {
    theblas::dtbsv(uplo, trans, diag, n, k, a, lda, x, incx);
}
void theblas_ctbsv(char uplo, char trans, char diag, int n, int k, const float *a, int lda,
                   float *x, int incx) {
    theblas::ctbsv(uplo, trans, diag, n, k, cplx(a), lda, cplx(x), incx);
}
void theblas_ztbsv(char uplo, char trans, char diag, int n, int k, const double *a, int lda,
                   double *x, int incx) {
    theblas::ztbsv(uplo, trans, diag, n, k, cplx(a), lda, cplx(x), incx);
}
void theblas_sspmv(char uplo, int n, float alpha, const float *ap, const float *x, int incx,
                   float beta, float *y, int incy) {
    theblas::sspmv(uplo, n, alpha, ap, x, incx, beta, y, incy);
}
void theblas_dspmv(char uplo, int n, double alpha, const double *ap, const double *x, int incx,
                   double beta, double *y, int incy) {
    theblas::dspmv(uplo, n, alpha, ap, x, incx, beta, y, incy);
}
void theblas_chpmv(char uplo, int n, const float *alpha, const float *ap, const float *x, int incx,
                   const float *beta, float *y, int incy) {
    theblas::chpmv(uplo, n, scalar(alpha), cplx(ap), cplx(x), incx, scalar(beta), cplx(y), incy);
}
void theblas_zhpmv(char uplo, int n, const double *alpha, const double *ap, const double *x,
                   int incx, const double *beta, double *y, int incy) {
    theblas::zhpmv(uplo, n, scalar(alpha), cplx(ap), cplx(x), incx, scalar(beta), cplx(y), incy);
}
void theblas_stpmv(char uplo, char trans, char diag, int n, const float *ap, float *x, int incx) {
    theblas::stpmv(uplo, trans, diag, n, ap, x, incx);
}
void theblas_dtpmv(char uplo, char trans, char diag, int n, const double *ap, double *x, int incx) {
    theblas::dtpmv(uplo, trans, diag, n, ap, x, incx);
}
void theblas_ctpmv(char uplo, char trans, char diag, int n, const float *ap, float *x, int incx) {
    theblas::ctpmv(uplo, trans, diag, n, cplx(ap), cplx(x), incx);
}
void theblas_ztpmv(char uplo, char trans, char diag, int n, const double *ap, double *x, int incx) {
    theblas::ztpmv(uplo, trans, diag, n, cplx(ap), cplx(x), incx);
}
void theblas_stpsv(char uplo, char trans, char diag, int n, const float *ap, float *x, int incx) {
    theblas::stpsv(uplo, trans, diag, n, ap, x, incx);
}
void theblas_dtpsv(char uplo, char trans, char diag, int n, const double *ap, double *x, int incx) {
    theblas::dtpsv(uplo, trans, diag, n, ap, x, incx);
}
void theblas_ctpsv(char uplo, char trans, char diag, int n, const float *ap, float *x, int incx) {
    theblas::ctpsv(uplo, trans, diag, n, cplx(ap), cplx(x), incx);
}
void theblas_ztpsv(char uplo, char trans, char diag, int n, const double *ap, double *x, int incx) {
    theblas::ztpsv(uplo, trans, diag, n, cplx(ap), cplx(x), incx);
}
void theblas_sspr(char uplo, int n, float alpha, const float *x, int incx, float *ap) {
    theblas::sspr(uplo, n, alpha, x, incx, ap);
}
void theblas_dspr(char uplo, int n, double alpha, const double *x, int incx, double *ap) {
    theblas::dspr(uplo, n, alpha, x, incx, ap);
}
void theblas_chpr(char uplo, int n, float alpha, const float *x, int incx, float *ap) {
    theblas::chpr(uplo, n, alpha, cplx(x), incx, cplx(ap));
}
void theblas_zhpr(char uplo, int n, double alpha, const double *x, int incx, double *ap) {
    theblas::zhpr(uplo, n, alpha, cplx(x), incx, cplx(ap));
}
void theblas_sspr2(char uplo, int n, float alpha, const float *x, int incx, const float *y,
                   int incy, float *ap) {
    theblas::sspr2(uplo, n, alpha, x, incx, y, incy, ap);
}
void theblas_dspr2(char uplo, int n, double alpha, const double *x, int incx, const double *y,
                   int incy, double *ap) {
    theblas::dspr2(uplo, n, alpha, x, incx, y, incy, ap);
}
void theblas_chpr2(char uplo, int n, const float *alpha, const float *x, int incx, const float *y,
                   int incy, float *ap) {
    theblas::chpr2(uplo, n, scalar(alpha), cplx(x), incx, cplx(y), incy, cplx(ap));
}
void theblas_zhpr2(char uplo, int n, const double *alpha, const double *x, int incx,
                   const double *y, int incy, double *ap) {
    theblas::zhpr2(uplo, n, scalar(alpha), cplx(x), incx, cplx(y), incy, cplx(ap));
}

// Level-3 matrix-matrix operations
void theblas_sgemm(char transa, char transb, int m, int n, int k, float alpha, const float *a,
                   int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    theblas::sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void theblas_dgemm(char transa, char transb, int m, int n, int k, double alpha, const double *a,
                   int lda, const double *b, int ldb, double beta, double *c, int ldc) {
    theblas::dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void theblas_cgemm(char transa, char transb, int m, int n, int k, const float *alpha,
                   const float *a, int lda, const float *b, int ldb, const float *beta, float *c,
                   int ldc) {
    theblas::cgemm(transa, transb, m, n, k, scalar(alpha), cplx(a), lda, cplx(b), ldb, scalar(beta),
                   cplx(c), ldc);
}
void theblas_zgemm(char transa, char transb, int m, int n, int k, const double *alpha,
                   const double *a, int lda, const double *b, int ldb, const double *beta,
                   double *c, int ldc) {
    theblas::zgemm(transa, transb, m, n, k, scalar(alpha), cplx(a), lda, cplx(b), ldb, scalar(beta),
                   cplx(c), ldc);
}
void theblas_ssymm(char side, char uplo, int m, int n, float alpha, const float *a, int lda,
                   const float *b, int ldb, float beta, float *c, int ldc) {
    theblas::ssymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
void theblas_dsymm(char side, char uplo, int m, int n, double alpha, const double *a, int lda,
                   const double *b, int ldb, double beta, double *c, int ldc) {
    theblas::dsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
void theblas_csymm(char side, char uplo, int m, int n, const float *alpha, const float *a, int lda,
                   const float *b, int ldb, const float *beta, float *c, int ldc) {
    theblas::csymm(side, uplo, m, n, scalar(alpha), cplx(a), lda, cplx(b), ldb, scalar(beta),
                   cplx(c), ldc);
}
void theblas_zsymm(char side, char uplo, int m, int n, const double *alpha, const double *a,
                   int lda, const double *b, int ldb, const double *beta, double *c, int ldc) {
    theblas::zsymm(side, uplo, m, n, scalar(alpha), cplx(a), lda, cplx(b), ldb, scalar(beta),
                   cplx(c), ldc);
}
void theblas_chemm(char side, char uplo, int m, int n, const float *alpha, const float *a, int lda,
                   const float *b, int ldb, const float *beta, float *c, int ldc) {
    theblas::chemm(side, uplo, m, n, scalar(alpha), cplx(a), lda, cplx(b), ldb, scalar(beta),
                   cplx(c), ldc);
}
void theblas_zhemm(char side, char uplo, int m, int n, const double *alpha, const double *a,
                   int lda, const double *b, int ldb, const double *beta, double *c, int ldc) {
    theblas::zhemm(side, uplo, m, n, scalar(alpha), cplx(a), lda, cplx(b), ldb, scalar(beta),
                   cplx(c), ldc);
}
void theblas_ssyrk(char uplo, char trans, int n, int k, float alpha, const float *a, int lda,
                   float beta, float *c, int ldc) {
    theblas::ssyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
void theblas_dsyrk(char uplo, char trans, int n, int k, double alpha, const double *a, int lda,
                   double beta, double *c, int ldc) {
    theblas::dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
void theblas_csyrk(char uplo, char trans, int n, int k, const float *alpha, const float *a, int lda,
                   const float *beta, float *c, int ldc) {
    theblas::csyrk(uplo, trans, n, k, scalar(alpha), cplx(a), lda, scalar(beta), cplx(c), ldc);
}
void theblas_zsyrk(char uplo, char trans, int n, int k, const double *alpha, const double *a,
                   int lda, const double *beta, double *c, int ldc) {
    theblas::zsyrk(uplo, trans, n, k, scalar(alpha), cplx(a), lda, scalar(beta), cplx(c), ldc);
}
void theblas_cherk(char uplo, char trans, int n, int k, float alpha, const float *a, int lda,
                   float beta, float *c, int ldc) {
    theblas::cherk(uplo, trans, n, k, alpha, cplx(a), lda, beta, cplx(c), ldc);
}
void theblas_zherk(char uplo, char trans, int n, int k, double alpha, const double *a, int lda,
                   double beta, double *c, int ldc) {
    theblas::zherk(uplo, trans, n, k, alpha, cplx(a), lda, beta, cplx(c), ldc);
}
void theblas_ssyr2k(char uplo, char trans, int n, int k, float alpha, const float *a, int lda,
                    const float *b, int ldb, float beta, float *c, int ldc) {
    theblas::ssyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void theblas_dsyr2k(char uplo, char trans, int n, int k, double alpha, const double *a, int lda,
                    const double *b, int ldb, double beta, double *c, int ldc) {
    theblas::dsyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void theblas_csyr2k(char uplo, char trans, int n, int k, const float *alpha, const float *a,
                    int lda, const float *b, int ldb, const float *beta, float *c, int ldc) {
    theblas::csyr2k(uplo, trans, n, k, scalar(alpha), cplx(a), lda, cplx(b), ldb, scalar(beta),
                    cplx(c), ldc);
}
void theblas_zsyr2k(char uplo, char trans, int n, int k, const double *alpha, const double *a,
                    int lda, const double *b, int ldb, const double *beta, double *c, int ldc) {
    theblas::zsyr2k(uplo, trans, n, k, scalar(alpha), cplx(a), lda, cplx(b), ldb, scalar(beta),
                    cplx(c), ldc);
}
void theblas_cher2k(char uplo, char trans, int n, int k, const float *alpha, const float *a,
                    int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    theblas::cher2k(uplo, trans, n, k, scalar(alpha), cplx(a), lda, cplx(b), ldb, beta, cplx(c),
                    ldc);
}
void theblas_zher2k(char uplo, char trans, int n, int k, const double *alpha, const double *a,
                    int lda, const double *b, int ldb, double beta, double *c, int ldc) {
    theblas::zher2k(uplo, trans, n, k, scalar(alpha), cplx(a), lda, cplx(b), ldb, beta, cplx(c),
                    ldc);
}
void theblas_strmm(char side, char uplo, char trans, char diag, int m, int n, float alpha,
                   const float *a, int lda, float *b, int ldb) {
    theblas::strmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
}
void theblas_dtrmm(char side, char uplo, char trans, char diag, int m, int n, double alpha,
                   const double *a, int lda, double *b, int ldb) {
    theblas::dtrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
}
void theblas_ctrmm(char side, char uplo, char trans, char diag, int m, int n, const float *alpha,
                   const float *a, int lda, float *b, int ldb) {
    theblas::ctrmm(side, uplo, trans, diag, m, n, scalar(alpha), cplx(a), lda, cplx(b), ldb);
}
void theblas_ztrmm(char side, char uplo, char trans, char diag, int m, int n, const double *alpha,
                   const double *a, int lda, double *b, int ldb) {
    theblas::ztrmm(side, uplo, trans, diag, m, n, scalar(alpha), cplx(a), lda, cplx(b), ldb);
}
void theblas_strsm(char side, char uplo, char trans, char diag, int m, int n, float alpha,
                   const float *a, int lda, float *b, int ldb) {
    theblas::strsm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
}
void theblas_dtrsm(char side, char uplo, char trans, char diag, int m, int n, double alpha,
                   const double *a, int lda, double *b, int ldb) {
    theblas::dtrsm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
}
void theblas_ctrsm(char side, char uplo, char trans, char diag, int m, int n, const float *alpha,
                   const float *a, int lda, float *b, int ldb) {
    theblas::ctrsm(side, uplo, trans, diag, m, n, scalar(alpha), cplx(a), lda, cplx(b), ldb);
}
void theblas_ztrsm(char side, char uplo, char trans, char diag, int m, int n, const double *alpha,
                   const double *a, int lda, double *b, int ldb) {
    theblas::ztrsm(side, uplo, trans, diag, m, n, scalar(alpha), cplx(a), lda, cplx(b), ldb);
}

} // extern "C"
