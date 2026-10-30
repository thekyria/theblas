#ifndef THEBLAS_THEBLAS_C_H
#define THEBLAS_THEBLAS_C_H

/**
 * @file theblas_c.h
 * @brief C ABI for theblas, intended for foreign-function interfaces (ctypes, cffi, ...).
 *
 * Every routine forwards to the C++ function of the same name in `theblas.h`, with the
 * `theblas_` prefix and identical argument order and semantics (column-major storage,
 * BLAS-style strides, 1-based `i?amax` results). The prefix avoids clashes with any
 * CBLAS/Fortran BLAS loaded into the same process.
 *
 * Complex values use interleaved `{real, imag}` storage:
 * - complex arrays are passed as `float *` / `double *` pointing to interleaved pairs;
 * - complex scalars (for example `alpha`) are passed as a pointer to one `{real, imag}` pair;
 * - complex return values (`?dotu`, `?dotc`) are written to a trailing `result` pair.
 */

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(THEBLAS_C_EXPORTS)
#define THEBLAS_C_API __declspec(dllexport)
#else
#define THEBLAS_C_API __declspec(dllimport)
#endif
#else
#define THEBLAS_C_API __attribute__((visibility("default")))
#endif

/** Argument error handler: Netlib-style routine name and 1-based parameter number. */
/* NOLINTNEXTLINE(modernize-use-using) */
typedef void (*theblas_error_handler_t)(const char *routine, int param);

/** Set the process-global Level-2/3 argument error handler; NULL restores the default. */
THEBLAS_C_API theblas_error_handler_t theblas_set_error_handler(theblas_error_handler_t handler);

/* Level-1 vector operations */
THEBLAS_C_API void theblas_sswap(int n, float *x, int incx, float *y, int incy);
THEBLAS_C_API void theblas_dswap(int n, double *x, int incx, double *y, int incy);
THEBLAS_C_API void theblas_cswap(int n, float *x, int incx, float *y, int incy);
THEBLAS_C_API void theblas_zswap(int n, double *x, int incx, double *y, int incy);
THEBLAS_C_API void theblas_scopy(int n, const float *x, int incx, float *y, int incy);
THEBLAS_C_API void theblas_dcopy(int n, const double *x, int incx, double *y, int incy);
THEBLAS_C_API void theblas_ccopy(int n, const float *x, int incx, float *y, int incy);
THEBLAS_C_API void theblas_zcopy(int n, const double *x, int incx, double *y, int incy);
THEBLAS_C_API void theblas_saxpy(int n, float alpha, const float *x, int incx, float *y, int incy);
THEBLAS_C_API void theblas_daxpy(int n, double alpha, const double *x, int incx, double *y,
                                 int incy);
THEBLAS_C_API void theblas_caxpy(int n, const float *alpha, const float *x, int incx, float *y,
                                 int incy);
THEBLAS_C_API void theblas_zaxpy(int n, const double *alpha, const double *x, int incx, double *y,
                                 int incy);
THEBLAS_C_API void theblas_sscal(int n, float alpha, float *x, int incx);
THEBLAS_C_API void theblas_dscal(int n, double alpha, double *x, int incx);
THEBLAS_C_API void theblas_cscal(int n, const float *alpha, float *x, int incx);
THEBLAS_C_API void theblas_zscal(int n, const double *alpha, double *x, int incx);
THEBLAS_C_API void theblas_csscal(int n, float alpha, float *x, int incx);
THEBLAS_C_API void theblas_zdscal(int n, double alpha, double *x, int incx);
THEBLAS_C_API float theblas_sdot(int n, const float *x, int incx, const float *y, int incy);
THEBLAS_C_API double theblas_ddot(int n, const double *x, int incx, const double *y, int incy);
THEBLAS_C_API void theblas_cdotu(int n, const float *x, int incx, const float *y, int incy,
                                 float *result);
THEBLAS_C_API void theblas_cdotc(int n, const float *x, int incx, const float *y, int incy,
                                 float *result);
THEBLAS_C_API void theblas_zdotu(int n, const double *x, int incx, const double *y, int incy,
                                 double *result);
THEBLAS_C_API void theblas_zdotc(int n, const double *x, int incx, const double *y, int incy,
                                 double *result);
THEBLAS_C_API float theblas_snrm2(int n, const float *x, int incx);
THEBLAS_C_API double theblas_dnrm2(int n, const double *x, int incx);
THEBLAS_C_API float theblas_scnrm2(int n, const float *x, int incx);
THEBLAS_C_API double theblas_dznrm2(int n, const double *x, int incx);
THEBLAS_C_API float theblas_sasum(int n, const float *x, int incx);
THEBLAS_C_API double theblas_dasum(int n, const double *x, int incx);
THEBLAS_C_API float theblas_scasum(int n, const float *x, int incx);
THEBLAS_C_API double theblas_dzasum(int n, const double *x, int incx);
THEBLAS_C_API void theblas_srot(int n, float *x, int incx, float *y, int incy, float c, float s);
THEBLAS_C_API void theblas_drot(int n, double *x, int incx, double *y, int incy, double c,
                                double s);
THEBLAS_C_API void theblas_csrot(int n, float *x, int incx, float *y, int incy, float c, float s);
THEBLAS_C_API void theblas_zdrot(int n, double *x, int incx, double *y, int incy, double c,
                                 double s);
THEBLAS_C_API void theblas_srotg(float *a, float *b, float *c, float *s);
THEBLAS_C_API void theblas_drotg(double *a, double *b, double *c, double *s);
THEBLAS_C_API void theblas_crotg(float *a, const float *b, float *c, float *s);
THEBLAS_C_API void theblas_zrotg(double *a, const double *b, double *c, double *s);
THEBLAS_C_API void theblas_srotm(int n, float *x, int incx, float *y, int incy, const float *param);
THEBLAS_C_API void theblas_drotm(int n, double *x, int incx, double *y, int incy,
                                 const double *param);
THEBLAS_C_API void theblas_srotmg(float *d1, float *d2, float *b1, float b2, float *param);
THEBLAS_C_API void theblas_drotmg(double *d1, double *d2, double *b1, double b2, double *param);
THEBLAS_C_API int theblas_isamax(int n, const float *x, int incx);
THEBLAS_C_API int theblas_idamax(int n, const double *x, int incx);
THEBLAS_C_API int theblas_icamax(int n, const float *x, int incx);
THEBLAS_C_API int theblas_izamax(int n, const double *x, int incx);

/* Level-2 matrix-vector operations */
THEBLAS_C_API void theblas_sgemv(char trans, int m, int n, float alpha, const float *a, int lda,
                                 const float *x, int incx, float beta, float *y, int incy);
THEBLAS_C_API void theblas_dgemv(char trans, int m, int n, double alpha, const double *a, int lda,
                                 const double *x, int incx, double beta, double *y, int incy);
THEBLAS_C_API void theblas_cgemv(char trans, int m, int n, const float *alpha, const float *a,
                                 int lda, const float *x, int incx, const float *beta, float *y,
                                 int incy);
THEBLAS_C_API void theblas_zgemv(char trans, int m, int n, const double *alpha, const double *a,
                                 int lda, const double *x, int incx, const double *beta, double *y,
                                 int incy);
THEBLAS_C_API void theblas_ssymv(char uplo, int n, float alpha, const float *a, int lda,
                                 const float *x, int incx, float beta, float *y, int incy);
THEBLAS_C_API void theblas_dsymv(char uplo, int n, double alpha, const double *a, int lda,
                                 const double *x, int incx, double beta, double *y, int incy);
THEBLAS_C_API void theblas_chemv(char uplo, int n, const float *alpha, const float *a, int lda,
                                 const float *x, int incx, const float *beta, float *y, int incy);
THEBLAS_C_API void theblas_zhemv(char uplo, int n, const double *alpha, const double *a, int lda,
                                 const double *x, int incx, const double *beta, double *y,
                                 int incy);
THEBLAS_C_API void theblas_strmv(char uplo, char trans, char diag, int n, const float *a, int lda,
                                 float *x, int incx);
THEBLAS_C_API void theblas_dtrmv(char uplo, char trans, char diag, int n, const double *a, int lda,
                                 double *x, int incx);
THEBLAS_C_API void theblas_ctrmv(char uplo, char trans, char diag, int n, const float *a, int lda,
                                 float *x, int incx);
THEBLAS_C_API void theblas_ztrmv(char uplo, char trans, char diag, int n, const double *a, int lda,
                                 double *x, int incx);
THEBLAS_C_API void theblas_strsv(char uplo, char trans, char diag, int n, const float *a, int lda,
                                 float *x, int incx);
THEBLAS_C_API void theblas_dtrsv(char uplo, char trans, char diag, int n, const double *a, int lda,
                                 double *x, int incx);
THEBLAS_C_API void theblas_ctrsv(char uplo, char trans, char diag, int n, const float *a, int lda,
                                 float *x, int incx);
THEBLAS_C_API void theblas_ztrsv(char uplo, char trans, char diag, int n, const double *a, int lda,
                                 double *x, int incx);
THEBLAS_C_API void theblas_sger(int m, int n, float alpha, const float *x, int incx, const float *y,
                                int incy, float *a, int lda);
THEBLAS_C_API void theblas_dger(int m, int n, double alpha, const double *x, int incx,
                                const double *y, int incy, double *a, int lda);
THEBLAS_C_API void theblas_cgeru(int m, int n, const float *alpha, const float *x, int incx,
                                 const float *y, int incy, float *a, int lda);
THEBLAS_C_API void theblas_zgeru(int m, int n, const double *alpha, const double *x, int incx,
                                 const double *y, int incy, double *a, int lda);
THEBLAS_C_API void theblas_cgerc(int m, int n, const float *alpha, const float *x, int incx,
                                 const float *y, int incy, float *a, int lda);
THEBLAS_C_API void theblas_zgerc(int m, int n, const double *alpha, const double *x, int incx,
                                 const double *y, int incy, double *a, int lda);
THEBLAS_C_API void theblas_ssyr(char uplo, int n, float alpha, const float *x, int incx, float *a,
                                int lda);
THEBLAS_C_API void theblas_dsyr(char uplo, int n, double alpha, const double *x, int incx,
                                double *a, int lda);
THEBLAS_C_API void theblas_cher(char uplo, int n, float alpha, const float *x, int incx, float *a,
                                int lda);
THEBLAS_C_API void theblas_zher(char uplo, int n, double alpha, const double *x, int incx,
                                double *a, int lda);
THEBLAS_C_API void theblas_ssyr2(char uplo, int n, float alpha, const float *x, int incx,
                                 const float *y, int incy, float *a, int lda);
THEBLAS_C_API void theblas_dsyr2(char uplo, int n, double alpha, const double *x, int incx,
                                 const double *y, int incy, double *a, int lda);
THEBLAS_C_API void theblas_cher2(char uplo, int n, const float *alpha, const float *x, int incx,
                                 const float *y, int incy, float *a, int lda);
THEBLAS_C_API void theblas_zher2(char uplo, int n, const double *alpha, const double *x, int incx,
                                 const double *y, int incy, double *a, int lda);
THEBLAS_C_API void theblas_sgbmv(char trans, int m, int n, int kl, int ku, float alpha,
                                 const float *a, int lda, const float *x, int incx, float beta,
                                 float *y, int incy);
THEBLAS_C_API void theblas_dgbmv(char trans, int m, int n, int kl, int ku, double alpha,
                                 const double *a, int lda, const double *x, int incx, double beta,
                                 double *y, int incy);
THEBLAS_C_API void theblas_cgbmv(char trans, int m, int n, int kl, int ku, const float *alpha,
                                 const float *a, int lda, const float *x, int incx,
                                 const float *beta, float *y, int incy);
THEBLAS_C_API void theblas_zgbmv(char trans, int m, int n, int kl, int ku, const double *alpha,
                                 const double *a, int lda, const double *x, int incx,
                                 const double *beta, double *y, int incy);
THEBLAS_C_API void theblas_ssbmv(char uplo, int n, int k, float alpha, const float *a, int lda,
                                 const float *x, int incx, float beta, float *y, int incy);
THEBLAS_C_API void theblas_dsbmv(char uplo, int n, int k, double alpha, const double *a, int lda,
                                 const double *x, int incx, double beta, double *y, int incy);
THEBLAS_C_API void theblas_chbmv(char uplo, int n, int k, const float *alpha, const float *a,
                                 int lda, const float *x, int incx, const float *beta, float *y,
                                 int incy);
THEBLAS_C_API void theblas_zhbmv(char uplo, int n, int k, const double *alpha, const double *a,
                                 int lda, const double *x, int incx, const double *beta, double *y,
                                 int incy);
THEBLAS_C_API void theblas_stbmv(char uplo, char trans, char diag, int n, int k, const float *a,
                                 int lda, float *x, int incx);
THEBLAS_C_API void theblas_dtbmv(char uplo, char trans, char diag, int n, int k, const double *a,
                                 int lda, double *x, int incx);
THEBLAS_C_API void theblas_ctbmv(char uplo, char trans, char diag, int n, int k, const float *a,
                                 int lda, float *x, int incx);
THEBLAS_C_API void theblas_ztbmv(char uplo, char trans, char diag, int n, int k, const double *a,
                                 int lda, double *x, int incx);
THEBLAS_C_API void theblas_stbsv(char uplo, char trans, char diag, int n, int k, const float *a,
                                 int lda, float *x, int incx);
THEBLAS_C_API void theblas_dtbsv(char uplo, char trans, char diag, int n, int k, const double *a,
                                 int lda, double *x, int incx);
THEBLAS_C_API void theblas_ctbsv(char uplo, char trans, char diag, int n, int k, const float *a,
                                 int lda, float *x, int incx);
THEBLAS_C_API void theblas_ztbsv(char uplo, char trans, char diag, int n, int k, const double *a,
                                 int lda, double *x, int incx);
THEBLAS_C_API void theblas_sspmv(char uplo, int n, float alpha, const float *ap, const float *x,
                                 int incx, float beta, float *y, int incy);
THEBLAS_C_API void theblas_dspmv(char uplo, int n, double alpha, const double *ap, const double *x,
                                 int incx, double beta, double *y, int incy);
THEBLAS_C_API void theblas_chpmv(char uplo, int n, const float *alpha, const float *ap,
                                 const float *x, int incx, const float *beta, float *y, int incy);
THEBLAS_C_API void theblas_zhpmv(char uplo, int n, const double *alpha, const double *ap,
                                 const double *x, int incx, const double *beta, double *y,
                                 int incy);
THEBLAS_C_API void theblas_stpmv(char uplo, char trans, char diag, int n, const float *ap, float *x,
                                 int incx);
THEBLAS_C_API void theblas_dtpmv(char uplo, char trans, char diag, int n, const double *ap,
                                 double *x, int incx);
THEBLAS_C_API void theblas_ctpmv(char uplo, char trans, char diag, int n, const float *ap, float *x,
                                 int incx);
THEBLAS_C_API void theblas_ztpmv(char uplo, char trans, char diag, int n, const double *ap,
                                 double *x, int incx);
THEBLAS_C_API void theblas_stpsv(char uplo, char trans, char diag, int n, const float *ap, float *x,
                                 int incx);
THEBLAS_C_API void theblas_dtpsv(char uplo, char trans, char diag, int n, const double *ap,
                                 double *x, int incx);
THEBLAS_C_API void theblas_ctpsv(char uplo, char trans, char diag, int n, const float *ap, float *x,
                                 int incx);
THEBLAS_C_API void theblas_ztpsv(char uplo, char trans, char diag, int n, const double *ap,
                                 double *x, int incx);
THEBLAS_C_API void theblas_sspr(char uplo, int n, float alpha, const float *x, int incx, float *ap);
THEBLAS_C_API void theblas_dspr(char uplo, int n, double alpha, const double *x, int incx,
                                double *ap);
THEBLAS_C_API void theblas_chpr(char uplo, int n, float alpha, const float *x, int incx, float *ap);
THEBLAS_C_API void theblas_zhpr(char uplo, int n, double alpha, const double *x, int incx,
                                double *ap);
THEBLAS_C_API void theblas_sspr2(char uplo, int n, float alpha, const float *x, int incx,
                                 const float *y, int incy, float *ap);
THEBLAS_C_API void theblas_dspr2(char uplo, int n, double alpha, const double *x, int incx,
                                 const double *y, int incy, double *ap);
THEBLAS_C_API void theblas_chpr2(char uplo, int n, const float *alpha, const float *x, int incx,
                                 const float *y, int incy, float *ap);
THEBLAS_C_API void theblas_zhpr2(char uplo, int n, const double *alpha, const double *x, int incx,
                                 const double *y, int incy, double *ap);

/* Level-3 matrix-matrix operations */
THEBLAS_C_API void theblas_sgemm(char transa, char transb, int m, int n, int k, float alpha,
                                 const float *a, int lda, const float *b, int ldb, float beta,
                                 float *c, int ldc);
THEBLAS_C_API void theblas_dgemm(char transa, char transb, int m, int n, int k, double alpha,
                                 const double *a, int lda, const double *b, int ldb, double beta,
                                 double *c, int ldc);
THEBLAS_C_API void theblas_cgemm(char transa, char transb, int m, int n, int k, const float *alpha,
                                 const float *a, int lda, const float *b, int ldb,
                                 const float *beta, float *c, int ldc);
THEBLAS_C_API void theblas_zgemm(char transa, char transb, int m, int n, int k, const double *alpha,
                                 const double *a, int lda, const double *b, int ldb,
                                 const double *beta, double *c, int ldc);
THEBLAS_C_API void theblas_ssymm(char side, char uplo, int m, int n, float alpha, const float *a,
                                 int lda, const float *b, int ldb, float beta, float *c, int ldc);
THEBLAS_C_API void theblas_dsymm(char side, char uplo, int m, int n, double alpha, const double *a,
                                 int lda, const double *b, int ldb, double beta, double *c,
                                 int ldc);
THEBLAS_C_API void theblas_csymm(char side, char uplo, int m, int n, const float *alpha,
                                 const float *a, int lda, const float *b, int ldb,
                                 const float *beta, float *c, int ldc);
THEBLAS_C_API void theblas_zsymm(char side, char uplo, int m, int n, const double *alpha,
                                 const double *a, int lda, const double *b, int ldb,
                                 const double *beta, double *c, int ldc);
THEBLAS_C_API void theblas_chemm(char side, char uplo, int m, int n, const float *alpha,
                                 const float *a, int lda, const float *b, int ldb,
                                 const float *beta, float *c, int ldc);
THEBLAS_C_API void theblas_zhemm(char side, char uplo, int m, int n, const double *alpha,
                                 const double *a, int lda, const double *b, int ldb,
                                 const double *beta, double *c, int ldc);
THEBLAS_C_API void theblas_ssyrk(char uplo, char trans, int n, int k, float alpha, const float *a,
                                 int lda, float beta, float *c, int ldc);
THEBLAS_C_API void theblas_dsyrk(char uplo, char trans, int n, int k, double alpha, const double *a,
                                 int lda, double beta, double *c, int ldc);
THEBLAS_C_API void theblas_csyrk(char uplo, char trans, int n, int k, const float *alpha,
                                 const float *a, int lda, const float *beta, float *c, int ldc);
THEBLAS_C_API void theblas_zsyrk(char uplo, char trans, int n, int k, const double *alpha,
                                 const double *a, int lda, const double *beta, double *c, int ldc);
THEBLAS_C_API void theblas_cherk(char uplo, char trans, int n, int k, float alpha, const float *a,
                                 int lda, float beta, float *c, int ldc);
THEBLAS_C_API void theblas_zherk(char uplo, char trans, int n, int k, double alpha, const double *a,
                                 int lda, double beta, double *c, int ldc);
THEBLAS_C_API void theblas_ssyr2k(char uplo, char trans, int n, int k, float alpha, const float *a,
                                  int lda, const float *b, int ldb, float beta, float *c, int ldc);
THEBLAS_C_API void theblas_dsyr2k(char uplo, char trans, int n, int k, double alpha,
                                  const double *a, int lda, const double *b, int ldb, double beta,
                                  double *c, int ldc);
THEBLAS_C_API void theblas_csyr2k(char uplo, char trans, int n, int k, const float *alpha,
                                  const float *a, int lda, const float *b, int ldb,
                                  const float *beta, float *c, int ldc);
THEBLAS_C_API void theblas_zsyr2k(char uplo, char trans, int n, int k, const double *alpha,
                                  const double *a, int lda, const double *b, int ldb,
                                  const double *beta, double *c, int ldc);
THEBLAS_C_API void theblas_cher2k(char uplo, char trans, int n, int k, const float *alpha,
                                  const float *a, int lda, const float *b, int ldb, float beta,
                                  float *c, int ldc);
THEBLAS_C_API void theblas_zher2k(char uplo, char trans, int n, int k, const double *alpha,
                                  const double *a, int lda, const double *b, int ldb, double beta,
                                  double *c, int ldc);
THEBLAS_C_API void theblas_strmm(char side, char uplo, char trans, char diag, int m, int n,
                                 float alpha, const float *a, int lda, float *b, int ldb);
THEBLAS_C_API void theblas_dtrmm(char side, char uplo, char trans, char diag, int m, int n,
                                 double alpha, const double *a, int lda, double *b, int ldb);
THEBLAS_C_API void theblas_ctrmm(char side, char uplo, char trans, char diag, int m, int n,
                                 const float *alpha, const float *a, int lda, float *b, int ldb);
THEBLAS_C_API void theblas_ztrmm(char side, char uplo, char trans, char diag, int m, int n,
                                 const double *alpha, const double *a, int lda, double *b, int ldb);
THEBLAS_C_API void theblas_strsm(char side, char uplo, char trans, char diag, int m, int n,
                                 float alpha, const float *a, int lda, float *b, int ldb);
THEBLAS_C_API void theblas_dtrsm(char side, char uplo, char trans, char diag, int m, int n,
                                 double alpha, const double *a, int lda, double *b, int ldb);
THEBLAS_C_API void theblas_ctrsm(char side, char uplo, char trans, char diag, int m, int n,
                                 const float *alpha, const float *a, int lda, float *b, int ldb);
THEBLAS_C_API void theblas_ztrsm(char side, char uplo, char trans, char diag, int m, int n,
                                 const double *alpha, const double *a, int lda, double *b, int ldb);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* THEBLAS_THEBLAS_C_H */
