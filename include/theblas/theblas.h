// NOLINTNEXTLINE(portability-avoid-pragma-once)
#pragma once

#include "theblas/version.h"

#include <complex>
#include <cstddef>

namespace theblas {

/**
 * @file theblas.h
 * @brief Public C++17 API for a minimal Level-1, Level-2, and Level-3 BLAS-like library.
 *
 * Naming follows classic BLAS conventions:
 * - s*: single-precision real (`float`)
 * - d*: double-precision real (`double`)
 * - c*: single-precision complex (`std::complex<float>`)
 * - z*: double-precision complex (`std::complex<double>`)
 *
 * Parameter conventions used by all routines:
 * - `n`: number of logical vector elements to process
 * - `x`, `y`: input/output vectors
 * - `incx`, `incy`: element strides (can be negative unless otherwise noted)
 *
 * Behavior notes:
 * - If `n <= 0`, routines are no-ops (or return zero / index 0).
 * - If a stride is zero, most routines are treated as no-ops (or return zero).
 * - For `*amax` routines, `incx` must be strictly positive; otherwise 0 is returned.
 * - `*amax` return values use Netlib BLAS indexing (1-based index).
 *
 * Preconditions:
 * - For calls that actually process elements (`n > 0` and valid strides), pointers must
 *   reference sufficient valid storage according to `n` and stride.
 */

/**
 * @brief Signature for theblas argument error handlers.
 *
 * The handler receives the Netlib-style routine name (for example, `"DGEMV"`) and the
 * 1-based parameter number that had an illegal value.
 */
using error_handler_t = void (*)(const char *routine, int param);

/**
 * @brief Set the process-global argument error handler used by Level-2 and Level-3 routines.
 *
 * Passing `nullptr` restores the default silent handler. Changing the handler is not
 * thread-safe while other threads may call theblas routines concurrently.
 *
 * @param handler Replacement handler or `nullptr` for the default silent handler.
 * @return The previously installed handler, or `nullptr` if the default silent handler
 *         was active.
 */
error_handler_t set_error_handler(error_handler_t handler);

/** @defgroup level1_ops Level-1 Vector Operations
 *  @brief BLAS-like operations on strided vectors.
 *  @{
 */

/**
 * @brief Swap two float vectors element-wise.
 * @param n Number of elements to process.
 * @param x First vector, updated in place.
 * @param incx Stride between elements of x.
 * @param y Second vector, updated in place.
 * @param incy Stride between elements of y.
 */
void sswap(int n, float *x, int incx, float *y, int incy);
/**
 * @brief Swap two double vectors element-wise.
 * @param n Number of elements to process.
 * @param x First vector, updated in place.
 * @param incx Stride between elements of x.
 * @param y Second vector, updated in place.
 * @param incy Stride between elements of y.
 */
void dswap(int n, double *x, int incx, double *y, int incy);

/**
 * @brief Swap two complex-float vectors element-wise.
 * @param n Number of elements to process.
 * @param x First vector, updated in place.
 * @param incx Stride between elements of x.
 * @param y Second vector, updated in place.
 * @param incy Stride between elements of y.
 */
void cswap(int n, std::complex<float> *x, int incx, std::complex<float> *y, int incy);
/**
 * @brief Swap two complex-double vectors element-wise.
 * @param n Number of elements to process.
 * @param x First vector, updated in place.
 * @param incx Stride between elements of x.
 * @param y Second vector, updated in place.
 * @param incy Stride between elements of y.
 */
void zswap(int n, std::complex<double> *x, int incx, std::complex<double> *y, int incy);

/**
 * @brief Copy a float vector into another vector.
 * @param n Number of elements to process.
 * @param x Source vector.
 * @param incx Stride between elements of x.
 * @param y Destination vector.
 * @param incy Stride between elements of y.
 */
void scopy(int n, const float *x, int incx, float *y, int incy);
/**
 * @brief Copy a double vector into another vector.
 * @param n Number of elements to process.
 * @param x Source vector.
 * @param incx Stride between elements of x.
 * @param y Destination vector.
 * @param incy Stride between elements of y.
 */
void dcopy(int n, const double *x, int incx, double *y, int incy);

/**
 * @brief Copy a complex-float vector into another vector.
 * @param n Number of elements to process.
 * @param x Source vector.
 * @param incx Stride between elements of x.
 * @param y Destination vector.
 * @param incy Stride between elements of y.
 */
void ccopy(int n, const std::complex<float> *x, int incx, std::complex<float> *y, int incy);
/**
 * @brief Copy a complex-double vector into another vector.
 * @param n Number of elements to process.
 * @param x Source vector.
 * @param incx Stride between elements of x.
 * @param y Destination vector.
 * @param incy Stride between elements of y.
 */
void zcopy(int n, const std::complex<double> *x, int incx, std::complex<double> *y, int incy);

/**
 * @brief Compute y <- alpha * x + y for float vectors.
 * @param n Number of elements to process.
 * @param alpha Scalar multiplier.
 * @param x Input vector x.
 * @param incx Stride between elements of x.
 * @param y Input/output vector y.
 * @param incy Stride between elements of y.
 */
void saxpy(int n, float alpha, const float *x, int incx, float *y, int incy);
/**
 * @brief Compute y <- alpha * x + y for double vectors.
 * @param n Number of elements to process.
 * @param alpha Scalar multiplier.
 * @param x Input vector x.
 * @param incx Stride between elements of x.
 * @param y Input/output vector y.
 * @param incy Stride between elements of y.
 */
void daxpy(int n, double alpha, const double *x, int incx, double *y, int incy);

/**
 * @brief Compute y <- alpha * x + y for complex-float vectors.
 * @param n Number of elements to process.
 * @param alpha Scalar multiplier.
 * @param x Input vector x.
 * @param incx Stride between elements of x.
 * @param y Input/output vector y.
 * @param incy Stride between elements of y.
 */
void caxpy(int n, std::complex<float> alpha, const std::complex<float> *x, int incx,
           std::complex<float> *y, int incy);
/**
 * @brief Compute y <- alpha * x + y for complex-double vectors.
 * @param n Number of elements to process.
 * @param alpha Scalar multiplier.
 * @param x Input vector x.
 * @param incx Stride between elements of x.
 * @param y Input/output vector y.
 * @param incy Stride between elements of y.
 */
void zaxpy(int n, std::complex<double> alpha, const std::complex<double> *x, int incx,
           std::complex<double> *y, int incy);

/**
 * @brief Scale a float vector: x <- alpha * x.
 * @param n Number of elements to process.
 * @param alpha Scale factor.
 * @param x Vector to scale in place.
 * @param incx Stride between elements of x.
 */
void sscal(int n, float alpha, float *x, int incx);
/**
 * @brief Scale a double vector: x <- alpha * x.
 * @param n Number of elements to process.
 * @param alpha Scale factor.
 * @param x Vector to scale in place.
 * @param incx Stride between elements of x.
 */
void dscal(int n, double alpha, double *x, int incx);

/**
 * @brief Scale a complex-float vector by a complex scalar.
 * @param n Number of elements to process.
 * @param alpha Complex scale factor.
 * @param x Vector to scale in place.
 * @param incx Stride between elements of x.
 */
void cscal(int n, std::complex<float> alpha, std::complex<float> *x, int incx);
/**
 * @brief Scale a complex-double vector by a complex scalar.
 * @param n Number of elements to process.
 * @param alpha Complex scale factor.
 * @param x Vector to scale in place.
 * @param incx Stride between elements of x.
 */
void zscal(int n, std::complex<double> alpha, std::complex<double> *x, int incx);
/**
 * @brief Scale a complex-float vector by a real scalar.
 * @param n Number of elements to process.
 * @param alpha Real scale factor.
 * @param x Vector to scale in place.
 * @param incx Stride between elements of x.
 */
void csscal(int n, float alpha, std::complex<float> *x, int incx);
/**
 * @brief Scale a complex-double vector by a real scalar.
 * @param n Number of elements to process.
 * @param alpha Real scale factor.
 * @param x Vector to scale in place.
 * @param incx Stride between elements of x.
 */
void zdscal(int n, double alpha, std::complex<double> *x, int incx);

/**
 * @brief Dot product of two float vectors.
 * @param n Number of elements to process.
 * @param x First input vector.
 * @param incx Stride between elements of x.
 * @param y Second input vector.
 * @param incy Stride between elements of y.
 * @return Dot product value.
 */
float sdot(int n, const float *x, int incx, const float *y, int incy);
/**
 * @brief Dot product of two double vectors.
 * @param n Number of elements to process.
 * @param x First input vector.
 * @param incx Stride between elements of x.
 * @param y Second input vector.
 * @param incy Stride between elements of y.
 * @return Dot product value.
 */
double ddot(int n, const double *x, int incx, const double *y, int incy);

/**
 * @brief Complex float dot product without conjugation.
 * @param n Number of elements to process.
 * @param x First input vector.
 * @param incx Stride between elements of x.
 * @param y Second input vector.
 * @param incy Stride between elements of y.
 * @return Complex dot product value.
 */
std::complex<float> cdotu(int n, const std::complex<float> *x, int incx,
                          const std::complex<float> *y, int incy);
/**
 * @brief Complex float dot product with conjugated first argument.
 * @param n Number of elements to process.
 * @param x First input vector, conjugated in the product.
 * @param incx Stride between elements of x.
 * @param y Second input vector.
 * @param incy Stride between elements of y.
 * @return Complex dot product value.
 */
std::complex<float> cdotc(int n, const std::complex<float> *x, int incx,
                          const std::complex<float> *y, int incy);
/**
 * @brief Complex double dot product without conjugation.
 * @param n Number of elements to process.
 * @param x First input vector.
 * @param incx Stride between elements of x.
 * @param y Second input vector.
 * @param incy Stride between elements of y.
 * @return Complex dot product value.
 */
std::complex<double> zdotu(int n, const std::complex<double> *x, int incx,
                           const std::complex<double> *y, int incy);
/**
 * @brief Complex double dot product with conjugated first argument.
 * @param n Number of elements to process.
 * @param x First input vector, conjugated in the product.
 * @param incx Stride between elements of x.
 * @param y Second input vector.
 * @param incy Stride between elements of y.
 * @return Complex dot product value.
 */
std::complex<double> zdotc(int n, const std::complex<double> *x, int incx,
                           const std::complex<double> *y, int incy);

/**
 * @brief Euclidean norm of a float vector.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @return Euclidean norm of x.
 */
float snrm2(int n, const float *x, int incx);
/**
 * @brief Euclidean norm of a double vector.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @return Euclidean norm of x.
 */
double dnrm2(int n, const double *x, int incx);

/**
 * @brief Euclidean norm of a complex-float vector.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @return Euclidean norm of x.
 */
float scnrm2(int n, const std::complex<float> *x, int incx);
/**
 * @brief Euclidean norm of a complex-double vector.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @return Euclidean norm of x.
 */
double dznrm2(int n, const std::complex<double> *x, int incx);

/**
 * @brief Sum of absolute values for a float vector.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @return Sum of absolute values.
 */
float sasum(int n, const float *x, int incx);
/**
 * @brief Sum of absolute values for a double vector.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @return Sum of absolute values.
 */
double dasum(int n, const double *x, int incx);

/**
 * @brief Sum of |Re(x_i)| + |Im(x_i)| for complex-float vectors.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @return Sum of absolute component values.
 */
float scasum(int n, const std::complex<float> *x, int incx);
/**
 * @brief Sum of |Re(x_i)| + |Im(x_i)| for complex-double vectors.
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @return Sum of absolute component values.
 */
double dzasum(int n, const std::complex<double> *x, int incx);

/**
 * @brief Apply a real Givens rotation to float vectors x and y.
 * @param n Number of elements to process.
 * @param x Input/output vector x.
 * @param incx Stride between elements of x.
 * @param y Input/output vector y.
 * @param incy Stride between elements of y.
 * @param c Cosine-like rotation coefficient.
 * @param s Sine-like rotation coefficient.
 */
void srot(int n, float *x, int incx, float *y, int incy, float c, float s);
/**
 * @brief Apply a real Givens rotation to double vectors x and y.
 * @param n Number of elements to process.
 * @param x Input/output vector x.
 * @param incx Stride between elements of x.
 * @param y Input/output vector y.
 * @param incy Stride between elements of y.
 * @param c Cosine-like rotation coefficient.
 * @param s Sine-like rotation coefficient.
 */
void drot(int n, double *x, int incx, double *y, int incy, double c, double s);
/**
 * @brief Apply a real Givens rotation to complex-float vectors x and y.
 * @param n Number of elements to process.
 * @param x Input/output vector x.
 * @param incx Stride between elements of x.
 * @param y Input/output vector y.
 * @param incy Stride between elements of y.
 * @param c Real cosine-like rotation coefficient.
 * @param s Real sine-like rotation coefficient.
 */
void csrot(int n, std::complex<float> *x, int incx, std::complex<float> *y, int incy, float c,
           float s);
/**
 * @brief Apply a real Givens rotation to complex-double vectors x and y.
 * @param n Number of elements to process.
 * @param x Input/output vector x.
 * @param incx Stride between elements of x.
 * @param y Input/output vector y.
 * @param incy Stride between elements of y.
 * @param c Real cosine-like rotation coefficient.
 * @param s Real sine-like rotation coefficient.
 */
void zdrot(int n, std::complex<double> *x, int incx, std::complex<double> *y, int incy, double c,
           double s);

/**
 * @brief Construct float Givens rotation parameters in place.
 * @param a On input: first scalar; on output: rotation radius.
 * @param b On input: second scalar; on output: implementation-defined auxiliary value.
 * @param c Output cosine coefficient.
 * @param s Output sine coefficient.
 */
void srotg(float *a, float *b, float *c, float *s);
/**
 * @brief Construct double Givens rotation parameters in place.
 * @param a On input: first scalar; on output: rotation radius.
 * @param b On input: second scalar; on output: implementation-defined auxiliary value.
 * @param c Output cosine coefficient.
 * @param s Output sine coefficient.
 */
void drotg(double *a, double *b, double *c, double *s);
/**
 * @brief Construct complex-float Givens rotation parameters in place.
 * @param a On input: first scalar; on output: rotation radius-like value.
 * @param b Second scalar.
 * @param c Output real cosine coefficient.
 * @param s Output complex sine-like coefficient.
 */
void crotg(std::complex<float> *a, std::complex<float> b, float *c, std::complex<float> *s);
/**
 * @brief Construct complex-double Givens rotation parameters in place.
 * @param a On input: first scalar; on output: rotation radius-like value.
 * @param b Second scalar.
 * @param c Output real cosine coefficient.
 * @param s Output complex sine-like coefficient.
 */
void zrotg(std::complex<double> *a, std::complex<double> b, double *c, std::complex<double> *s);

/**
 * @brief Apply modified Givens rotation to float vectors using param.
 * @param n Number of elements to process.
 * @param x Input/output vector x.
 * @param incx Stride between elements of x.
 * @param y Input/output vector y.
 * @param incy Stride between elements of y.
 * @param param Pointer to a 5-element modified Givens parameter array.
 */
void srotm(int n, float *x, int incx, float *y, int incy, const float *param);
/**
 * @brief Apply modified Givens rotation to double vectors using param.
 * @param n Number of elements to process.
 * @param x Input/output vector x.
 * @param incx Stride between elements of x.
 * @param y Input/output vector y.
 * @param incy Stride between elements of y.
 * @param param Pointer to a 5-element modified Givens parameter array.
 */
void drotm(int n, double *x, int incx, double *y, int incy, const double *param);

/**
 * @brief Construct modified Givens parameters for float values.
 * @param d1 Scale factor component, updated in place.
 * @param d2 Scale factor component, updated in place.
 * @param b1 Input/output vector component.
 * @param b2 Input vector component.
 * @param param Output 5-element modified Givens parameter array.
 */
void srotmg(float *d1, float *d2, float *b1, float b2, float *param);
/**
 * @brief Construct modified Givens parameters for double values.
 * @param d1 Scale factor component, updated in place.
 * @param d2 Scale factor component, updated in place.
 * @param b1 Input/output vector component.
 * @param b2 Input vector component.
 * @param param Output 5-element modified Givens parameter array.
 */
void drotmg(double *d1, double *d2, double *b1, double b2, double *param);

/**
 * @brief Index of max absolute value in a float vector (1-based).
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x; must be positive.
 * @return Netlib-style 1-based index, or 0 when `n <= 0` or `incx <= 0`.
 */
int isamax(int n, const float *x, int incx);
/**
 * @brief Index of max absolute value in a double vector (1-based).
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x; must be positive.
 * @return Netlib-style 1-based index, or 0 when `n <= 0` or `incx <= 0`.
 */
int idamax(int n, const double *x, int incx);
/**
 * @brief Index of max absolute value in a complex-float vector (1-based).
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x; must be positive.
 * @return Netlib-style 1-based index, or 0 when `n <= 0` or `incx <= 0`.
 */
int icamax(int n, const std::complex<float> *x, int incx);
/**
 * @brief Index of max absolute value in a complex-double vector (1-based).
 * @param n Number of elements to process.
 * @param x Input vector.
 * @param incx Stride between elements of x; must be positive.
 * @return Netlib-style 1-based index, or 0 when `n <= 0` or `incx <= 0`.
 */
int izamax(int n, const std::complex<double> *x, int incx);

/** @} */

/** @defgroup level2_ops Level-2 Matrix-Vector Operations
 *  @brief BLAS-like matrix-vector operations with strided storage.
 *
 *  Level-2 routines operate on a matrix **a** and one or two vectors.
 *  Matrices are stored in **column-major** order (Fortran layout):
 *  element \f$a(i,j)\f$ resides at offset \f$i + j \cdot \text{lda}\f$.
 *
 *  **Additional parameter conventions:**
 *  - `lda`  — leading dimension of the matrix (≥ max(1, rows))
 *  - `trans`— `'N'` no-transpose, `'T'` transpose, `'C'` conjugate-transpose
 *  - `uplo` — `'U'` upper triangle, `'L'` lower triangle
 *  - `diag` — `'U'` unit diagonal, `'N'` non-unit diagonal
 *  - For banded matrices: `kl` sub-diagonals, `ku` super-diagonals, `k` bandwidth
 *  - Packed storage stores the upper or lower triangle as a contiguous array
 *    of \f$n(n+1)/2\f$ elements in column-major order.
 *
 *  Behaviour notes:
 *  - If `m <= 0` or `n <= 0`, routines are no-ops (or leave outputs unchanged).
 *  - Invalid `trans`, `uplo`, or `diag` characters cause early return (no-op).
 *
 *  @{
 */

/* ------------------------------------------------------------------ */
/* gemv — General Matrix-Vector Multiply                              */
/*   y ← α·op(a)·x + β·y                                            */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision general matrix-vector multiply: y ← α·op(a)·x + β·y.
 * @param trans 'N' no-transpose, 'T' transpose, 'C' conjugate-transpose.
 * @param m Number of rows of a.
 * @param n Number of columns of a.
 * @param alpha Scalar multiplier for op(a)·x.
 * @param a Pointer to the m×n matrix in column-major order.
 * @param lda Leading dimension of a (≥ max(1,m)).
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @param beta Scalar multiplier for y.
 * @param y Input/output vector.
 * @param incy Stride between elements of y.
 */
void sgemv(char trans, int m, int n, float alpha, const float *a, int lda, const float *x, int incx,
           float beta, float *y, int incy);
/** @copydoc sgemv */
void dgemv(char trans, int m, int n, double alpha, const double *a, int lda, const double *x,
           int incx, double beta, double *y, int incy);
/** @copydoc sgemv */
void cgemv(char trans, int m, int n, std::complex<float> alpha, const std::complex<float> *a,
           int lda, const std::complex<float> *x, int incx, std::complex<float> beta,
           std::complex<float> *y, int incy);
/** @copydoc sgemv */
void zgemv(char trans, int m, int n, std::complex<double> alpha, const std::complex<double> *a,
           int lda, const std::complex<double> *x, int incx, std::complex<double> beta,
           std::complex<double> *y, int incy);

/* ------------------------------------------------------------------ */
/* symv — Symmetric Matrix-Vector Multiply                            */
/*   y ← α·a·x + β·y   (a symmetric)                                */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision symmetric matrix-vector multiply: y ← α·a·x + β·y.
 * @param uplo 'U' upper triangle stored, 'L' lower triangle stored.
 * @param n Order of the symmetric matrix a.
 * @param alpha Scalar multiplier for a·x.
 * @param a Pointer to the n×n symmetric matrix in column-major order.
 * @param lda Leading dimension of a (≥ max(1,n)).
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @param beta Scalar multiplier for y.
 * @param y Input/output vector.
 * @param incy Stride between elements of y.
 */
void ssymv(char uplo, int n, float alpha, const float *a, int lda, const float *x, int incx,
           float beta, float *y, int incy);
/** @copydoc ssymv */
void dsymv(char uplo, int n, double alpha, const double *a, int lda, const double *x, int incx,
           double beta, double *y, int incy);

/* ------------------------------------------------------------------ */
/* hemv — Hermitian Matrix-Vector Multiply                            */
/*   y ← α·a·x + β·y   (a Hermitian)                                */
/* ------------------------------------------------------------------ */

/**
 * @brief Complex-float Hermitian matrix-vector multiply: y ← α·a·x + β·y.
 * @param uplo 'U' upper triangle stored, 'L' lower triangle stored.
 * @param n Order of the Hermitian matrix a.
 * @param alpha Scalar multiplier for a·x.
 * @param a Pointer to the n×n Hermitian matrix in column-major order.
 * @param lda Leading dimension of a (≥ max(1,n)).
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @param beta Scalar multiplier for y.
 * @param y Input/output vector.
 * @param incy Stride between elements of y.
 */
void chemv(char uplo, int n, std::complex<float> alpha, const std::complex<float> *a, int lda,
           const std::complex<float> *x, int incx, std::complex<float> beta, std::complex<float> *y,
           int incy);
/** @copydoc chemv */
void zhemv(char uplo, int n, std::complex<double> alpha, const std::complex<double> *a, int lda,
           const std::complex<double> *x, int incx, std::complex<double> beta,
           std::complex<double> *y, int incy);

/* ------------------------------------------------------------------ */
/* trmv — Triangular Matrix-Vector Multiply                           */
/*   x ← op(a)·x   (a triangular)                                   */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision triangular matrix-vector multiply: x ← op(a)·x.
 * @param uplo 'U' upper triangular, 'L' lower triangular.
 * @param trans 'N' no-transpose, 'T' transpose, 'C' conjugate-transpose.
 * @param diag 'U' unit diagonal, 'N' non-unit diagonal.
 * @param n Order of the triangular matrix a.
 * @param a Pointer to the n×n triangular matrix in column-major order.
 * @param lda Leading dimension of a (≥ max(1,n)).
 * @param x Input/output vector.
 * @param incx Stride between elements of x.
 */
void strmv(char uplo, char trans, char diag, int n, const float *a, int lda, float *x, int incx);
/** @copydoc strmv */
void dtrmv(char uplo, char trans, char diag, int n, const double *a, int lda, double *x, int incx);
/** @copydoc strmv */
void ctrmv(char uplo, char trans, char diag, int n, const std::complex<float> *a, int lda,
           std::complex<float> *x, int incx);
/** @copydoc strmv */
void ztrmv(char uplo, char trans, char diag, int n, const std::complex<double> *a, int lda,
           std::complex<double> *x, int incx);

/* ------------------------------------------------------------------ */
/* trsv — Triangular Solve                                            */
/*   solve op(a)·x = b   (a triangular, x overwrites b)              */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision triangular solve: solve op(a)·x = b.
 * @param uplo 'U' upper triangular, 'L' lower triangular.
 * @param trans 'N' no-transpose, 'T' transpose, 'C' conjugate-transpose.
 * @param diag 'U' unit diagonal, 'N' non-unit diagonal.
 * @param n Order of the triangular matrix a.
 * @param a Pointer to the n×n triangular matrix in column-major order.
 * @param lda Leading dimension of a (≥ max(1,n)).
 * @param x On entry, the right-hand side b; on exit, the solution x.
 * @param incx Stride between elements of x.
 */
void strsv(char uplo, char trans, char diag, int n, const float *a, int lda, float *x, int incx);
/** @copydoc strsv */
void dtrsv(char uplo, char trans, char diag, int n, const double *a, int lda, double *x, int incx);
/** @copydoc strsv */
void ctrsv(char uplo, char trans, char diag, int n, const std::complex<float> *a, int lda,
           std::complex<float> *x, int incx);
/** @copydoc strsv */
void ztrsv(char uplo, char trans, char diag, int n, const std::complex<double> *a, int lda,
           std::complex<double> *x, int incx);

/* ------------------------------------------------------------------ */
/* ger — General Rank-1 Update (real)                                 */
/*   a ← α·x·yᵀ + a                                                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision rank-1 update: a ← α·x·yᵀ + a.
 * @param m Number of rows of a.
 * @param n Number of columns of a.
 * @param alpha Scalar multiplier.
 * @param x Input vector of length m.
 * @param incx Stride between elements of x.
 * @param y Input vector of length n.
 * @param incy Stride between elements of y.
 * @param a Input/output m×n matrix in column-major order.
 * @param lda Leading dimension of a (≥ max(1,m)).
 */
void sger(int m, int n, float alpha, const float *x, int incx, const float *y, int incy, float *a,
          int lda);
/** @copydoc sger */
void dger(int m, int n, double alpha, const double *x, int incx, const double *y, int incy,
          double *a, int lda);

/* ------------------------------------------------------------------ */
/* geru / gerc — General Rank-1 Update (complex)                      */
/*   geru: a ← α·x·yᵀ + a   (unconjugated)                          */
/*   gerc: a ← α·x·yᴴ + a   (conjugated)                            */
/* ------------------------------------------------------------------ */

/**
 * @brief Complex-float unconjugated rank-1 update: a ← α·x·yᵀ + a.
 * @param m Number of rows of a.
 * @param n Number of columns of a.
 * @param alpha Scalar multiplier.
 * @param x Input vector of length m.
 * @param incx Stride between elements of x.
 * @param y Input vector of length n.
 * @param incy Stride between elements of y.
 * @param a Input/output m×n matrix in column-major order.
 * @param lda Leading dimension of a (≥ max(1,m)).
 */
void cgeru(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int incx,
           const std::complex<float> *y, int incy, std::complex<float> *a, int lda);
/** @copydoc cgeru */
void zgeru(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int incx,
           const std::complex<double> *y, int incy, std::complex<double> *a, int lda);

/**
 * @brief Complex-float conjugated rank-1 update: a ← α·x·yᴴ + a.
 * @param m Number of rows of a.
 * @param n Number of columns of a.
 * @param alpha Scalar multiplier.
 * @param x Input vector of length m.
 * @param incx Stride between elements of x.
 * @param y Input vector of length n (conjugated in the product).
 * @param incy Stride between elements of y.
 * @param a Input/output m×n matrix in column-major order.
 * @param lda Leading dimension of a (≥ max(1,m)).
 */
void cgerc(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int incx,
           const std::complex<float> *y, int incy, std::complex<float> *a, int lda);
/** @copydoc cgerc */
void zgerc(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int incx,
           const std::complex<double> *y, int incy, std::complex<double> *a, int lda);

/* ------------------------------------------------------------------ */
/* syr — Symmetric Rank-1 Update                                      */
/*   a ← α·x·xᵀ + a   (a symmetric)                                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision symmetric rank-1 update: a ← α·x·xᵀ + a.
 * @param uplo 'U' upper triangle stored, 'L' lower triangle stored.
 * @param n Order of the symmetric matrix a.
 * @param alpha Scalar multiplier.
 * @param x Input vector of length n.
 * @param incx Stride between elements of x.
 * @param a Input/output n×n symmetric matrix in column-major order.
 * @param lda Leading dimension of a (≥ max(1,n)).
 */
void ssyr(char uplo, int n, float alpha, const float *x, int incx, float *a, int lda);
/** @copydoc ssyr */
void dsyr(char uplo, int n, double alpha, const double *x, int incx, double *a, int lda);

/* ------------------------------------------------------------------ */
/* her — Hermitian Rank-1 Update                                      */
/*   a ← α·x·xᴴ + a   (a Hermitian, α real)                         */
/* ------------------------------------------------------------------ */

/**
 * @brief Complex-float Hermitian rank-1 update: a ← α·x·xᴴ + a.
 * @param uplo 'U' upper triangle stored, 'L' lower triangle stored.
 * @param n Order of the Hermitian matrix a.
 * @param alpha Real scalar multiplier.
 * @param x Input vector of length n.
 * @param incx Stride between elements of x.
 * @param a Input/output n×n Hermitian matrix in column-major order.
 * @param lda Leading dimension of a (≥ max(1,n)).
 */
void cher(char uplo, int n, float alpha, const std::complex<float> *x, int incx,
          std::complex<float> *a, int lda);
/** @copydoc cher */
void zher(char uplo, int n, double alpha, const std::complex<double> *x, int incx,
          std::complex<double> *a, int lda);

/* ------------------------------------------------------------------ */
/* syr2 — Symmetric Rank-2 Update                                     */
/*   a ← α·x·yᵀ + α·y·xᵀ + a   (a symmetric)                       */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision symmetric rank-2 update: a ← α·x·yᵀ + α·y·xᵀ + a.
 * @param uplo 'U' upper triangle stored, 'L' lower triangle stored.
 * @param n Order of the symmetric matrix a.
 * @param alpha Scalar multiplier.
 * @param x Input vector of length n.
 * @param incx Stride between elements of x.
 * @param y Input vector of length n.
 * @param incy Stride between elements of y.
 * @param a Input/output n×n symmetric matrix in column-major order.
 * @param lda Leading dimension of a (≥ max(1,n)).
 */
void ssyr2(char uplo, int n, float alpha, const float *x, int incx, const float *y, int incy,
           float *a, int lda);
/** @copydoc ssyr2 */
void dsyr2(char uplo, int n, double alpha, const double *x, int incx, const double *y, int incy,
           double *a, int lda);

/* ------------------------------------------------------------------ */
/* her2 — Hermitian Rank-2 Update                                     */
/*   a ← α·x·yᴴ + conj(α)·y·xᴴ + a   (a Hermitian)                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Complex-float Hermitian rank-2 update: a ← α·x·yᴴ + conj(α)·y·xᴴ + a.
 * @param uplo 'U' upper triangle stored, 'L' lower triangle stored.
 * @param n Order of the Hermitian matrix a.
 * @param alpha Complex scalar multiplier.
 * @param x Input vector of length n.
 * @param incx Stride between elements of x.
 * @param y Input vector of length n.
 * @param incy Stride between elements of y.
 * @param a Input/output n×n Hermitian matrix in column-major order.
 * @param lda Leading dimension of a (≥ max(1,n)).
 */
void cher2(char uplo, int n, std::complex<float> alpha, const std::complex<float> *x, int incx,
           const std::complex<float> *y, int incy, std::complex<float> *a, int lda);
/** @copydoc cher2 */
void zher2(char uplo, int n, std::complex<double> alpha, const std::complex<double> *x, int incx,
           const std::complex<double> *y, int incy, std::complex<double> *a, int lda);

/* ------------------------------------------------------------------ */
/* gbmv — General Band Matrix-Vector Multiply                         */
/*   y ← α·op(a)·x + β·y   (a banded)                               */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision general band matrix-vector multiply: y ← α·op(a)·x + β·y.
 * @param trans 'N' no-transpose, 'T' transpose, 'C' conjugate-transpose.
 * @param m Number of rows of a.
 * @param n Number of columns of a.
 * @param kl Number of sub-diagonals.
 * @param ku Number of super-diagonals.
 * @param alpha Scalar multiplier for op(a)·x.
 * @param a Pointer to the band matrix in column-major band storage.
 * @param lda Leading dimension of a (≥ kl+ku+1).
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @param beta Scalar multiplier for y.
 * @param y Input/output vector.
 * @param incy Stride between elements of y.
 */
void sgbmv(char trans, int m, int n, int kl, int ku, float alpha, const float *a, int lda,
           const float *x, int incx, float beta, float *y, int incy);
/** @copydoc sgbmv */
void dgbmv(char trans, int m, int n, int kl, int ku, double alpha, const double *a, int lda,
           const double *x, int incx, double beta, double *y, int incy);
/** @copydoc sgbmv */
void cgbmv(char trans, int m, int n, int kl, int ku, std::complex<float> alpha,
           const std::complex<float> *a, int lda, const std::complex<float> *x, int incx,
           std::complex<float> beta, std::complex<float> *y, int incy);
/** @copydoc sgbmv */
void zgbmv(char trans, int m, int n, int kl, int ku, std::complex<double> alpha,
           const std::complex<double> *a, int lda, const std::complex<double> *x, int incx,
           std::complex<double> beta, std::complex<double> *y, int incy);

/* ------------------------------------------------------------------ */
/* sbmv — Symmetric Band Matrix-Vector Multiply                       */
/*   y ← α·a·x + β·y   (a symmetric banded)                         */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision symmetric band matrix-vector multiply: y ← α·a·x + β·y.
 * @param uplo 'U' upper triangle stored, 'L' lower triangle stored.
 * @param n Order of the symmetric matrix a.
 * @param k Number of super-diagonals (bandwidth).
 * @param alpha Scalar multiplier for a·x.
 * @param a Pointer to the symmetric band matrix in column-major band storage.
 * @param lda Leading dimension of a (≥ k+1).
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @param beta Scalar multiplier for y.
 * @param y Input/output vector.
 * @param incy Stride between elements of y.
 */
void ssbmv(char uplo, int n, int k, float alpha, const float *a, int lda, const float *x, int incx,
           float beta, float *y, int incy);
/** @copydoc ssbmv */
void dsbmv(char uplo, int n, int k, double alpha, const double *a, int lda, const double *x,
           int incx, double beta, double *y, int incy);

/* ------------------------------------------------------------------ */
/* hbmv — Hermitian Band Matrix-Vector Multiply                       */
/*   y ← α·a·x + β·y   (a Hermitian banded)                         */
/* ------------------------------------------------------------------ */

/**
 * @brief Complex-float Hermitian band matrix-vector multiply: y ← α·a·x + β·y.
 * @param uplo 'U' upper triangle stored, 'L' lower triangle stored.
 * @param n Order of the Hermitian matrix a.
 * @param k Number of super-diagonals (bandwidth).
 * @param alpha Scalar multiplier for a·x.
 * @param a Pointer to the Hermitian band matrix in column-major band storage.
 * @param lda Leading dimension of a (≥ k+1).
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @param beta Scalar multiplier for y.
 * @param y Input/output vector.
 * @param incy Stride between elements of y.
 */
void chbmv(char uplo, int n, int k, std::complex<float> alpha, const std::complex<float> *a,
           int lda, const std::complex<float> *x, int incx, std::complex<float> beta,
           std::complex<float> *y, int incy);
/** @copydoc chbmv */
void zhbmv(char uplo, int n, int k, std::complex<double> alpha, const std::complex<double> *a,
           int lda, const std::complex<double> *x, int incx, std::complex<double> beta,
           std::complex<double> *y, int incy);

/* ------------------------------------------------------------------ */
/* tbmv — Triangular Band Matrix-Vector Multiply                      */
/*   x ← op(a)·x   (a triangular banded)                             */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision triangular band matrix-vector multiply: x ← op(a)·x.
 * @param uplo 'U' upper triangular, 'L' lower triangular.
 * @param trans 'N' no-transpose, 'T' transpose, 'C' conjugate-transpose.
 * @param diag 'U' unit diagonal, 'N' non-unit diagonal.
 * @param n Order of the triangular matrix a.
 * @param k Number of super-diagonals (uplo='U') or sub-diagonals (uplo='L').
 * @param a Pointer to the triangular band matrix in column-major band storage.
 * @param lda Leading dimension of a (≥ k+1).
 * @param x Input/output vector.
 * @param incx Stride between elements of x.
 */
void stbmv(char uplo, char trans, char diag, int n, int k, const float *a, int lda, float *x,
           int incx);
/** @copydoc stbmv */
void dtbmv(char uplo, char trans, char diag, int n, int k, const double *a, int lda, double *x,
           int incx);
/** @copydoc stbmv */
void ctbmv(char uplo, char trans, char diag, int n, int k, const std::complex<float> *a, int lda,
           std::complex<float> *x, int incx);
/** @copydoc stbmv */
void ztbmv(char uplo, char trans, char diag, int n, int k, const std::complex<double> *a, int lda,
           std::complex<double> *x, int incx);

/* ------------------------------------------------------------------ */
/* tbsv — Triangular Band Solve                                       */
/*   solve op(a)·x = b   (a triangular banded)                       */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision triangular band solve: solve op(a)·x = b.
 * @param uplo 'U' upper triangular, 'L' lower triangular.
 * @param trans 'N' no-transpose, 'T' transpose, 'C' conjugate-transpose.
 * @param diag 'U' unit diagonal, 'N' non-unit diagonal.
 * @param n Order of the triangular matrix a.
 * @param k Number of super-diagonals (uplo='U') or sub-diagonals (uplo='L').
 * @param a Pointer to the triangular band matrix in column-major band storage.
 * @param lda Leading dimension of a (≥ k+1).
 * @param x On entry, the right-hand side b; on exit, the solution x.
 * @param incx Stride between elements of x.
 */
void stbsv(char uplo, char trans, char diag, int n, int k, const float *a, int lda, float *x,
           int incx);
/** @copydoc stbsv */
void dtbsv(char uplo, char trans, char diag, int n, int k, const double *a, int lda, double *x,
           int incx);
/** @copydoc stbsv */
void ctbsv(char uplo, char trans, char diag, int n, int k, const std::complex<float> *a, int lda,
           std::complex<float> *x, int incx);
/** @copydoc stbsv */
void ztbsv(char uplo, char trans, char diag, int n, int k, const std::complex<double> *a, int lda,
           std::complex<double> *x, int incx);

/* ------------------------------------------------------------------ */
/* spmv — Symmetric Packed Matrix-Vector Multiply                     */
/*   y ← α·a·x + β·y   (a symmetric, packed storage)                */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision symmetric packed matrix-vector multiply: y ← α·a·x + β·y.
 * @param uplo 'U' upper triangle packed, 'L' lower triangle packed.
 * @param n Order of the symmetric matrix a.
 * @param alpha Scalar multiplier for a·x.
 * @param ap Packed storage array of n(n+1)/2 elements.
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @param beta Scalar multiplier for y.
 * @param y Input/output vector.
 * @param incy Stride between elements of y.
 */
void sspmv(char uplo, int n, float alpha, const float *ap, const float *x, int incx, float beta,
           float *y, int incy);
/** @copydoc sspmv */
void dspmv(char uplo, int n, double alpha, const double *ap, const double *x, int incx, double beta,
           double *y, int incy);

/* ------------------------------------------------------------------ */
/* hpmv — Hermitian Packed Matrix-Vector Multiply                     */
/*   y ← α·a·x + β·y   (a Hermitian, packed storage)                */
/* ------------------------------------------------------------------ */

/**
 * @brief Complex-float Hermitian packed matrix-vector multiply: y ← α·a·x + β·y.
 * @param uplo 'U' upper triangle packed, 'L' lower triangle packed.
 * @param n Order of the Hermitian matrix a.
 * @param alpha Scalar multiplier for a·x.
 * @param ap Packed storage array of n(n+1)/2 elements.
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @param beta Scalar multiplier for y.
 * @param y Input/output vector.
 * @param incy Stride between elements of y.
 */
void chpmv(char uplo, int n, std::complex<float> alpha, const std::complex<float> *ap,
           const std::complex<float> *x, int incx, std::complex<float> beta, std::complex<float> *y,
           int incy);
/** @copydoc chpmv */
void zhpmv(char uplo, int n, std::complex<double> alpha, const std::complex<double> *ap,
           const std::complex<double> *x, int incx, std::complex<double> beta,
           std::complex<double> *y, int incy);

/* ------------------------------------------------------------------ */
/* tpmv — Triangular Packed Matrix-Vector Multiply                    */
/*   x ← op(a)·x   (a triangular, packed storage)                   */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision triangular packed matrix-vector multiply: x ← op(a)·x.
 * @param uplo 'U' upper triangular packed, 'L' lower triangular packed.
 * @param trans 'N' no-transpose, 'T' transpose, 'C' conjugate-transpose.
 * @param diag 'U' unit diagonal, 'N' non-unit diagonal.
 * @param n Order of the triangular matrix a.
 * @param ap Packed storage array of n(n+1)/2 elements.
 * @param x Input/output vector.
 * @param incx Stride between elements of x.
 */
void stpmv(char uplo, char trans, char diag, int n, const float *ap, float *x, int incx);
/** @copydoc stpmv */
void dtpmv(char uplo, char trans, char diag, int n, const double *ap, double *x, int incx);
/** @copydoc stpmv */
void ctpmv(char uplo, char trans, char diag, int n, const std::complex<float> *ap,
           std::complex<float> *x, int incx);
/** @copydoc stpmv */
void ztpmv(char uplo, char trans, char diag, int n, const std::complex<double> *ap,
           std::complex<double> *x, int incx);

/* ------------------------------------------------------------------ */
/* tpsv — Triangular Packed Solve                                     */
/*   solve op(a)·x = b   (a triangular, packed storage)              */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision triangular packed solve: solve op(a)·x = b.
 * @param uplo 'U' upper triangular packed, 'L' lower triangular packed.
 * @param trans 'N' no-transpose, 'T' transpose, 'C' conjugate-transpose.
 * @param diag 'U' unit diagonal, 'N' non-unit diagonal.
 * @param n Order of the triangular matrix a.
 * @param ap Packed storage array of n(n+1)/2 elements.
 * @param x On entry, the right-hand side b; on exit, the solution x.
 * @param incx Stride between elements of x.
 */
void stpsv(char uplo, char trans, char diag, int n, const float *ap, float *x, int incx);
/** @copydoc stpsv */
void dtpsv(char uplo, char trans, char diag, int n, const double *ap, double *x, int incx);
/** @copydoc stpsv */
void ctpsv(char uplo, char trans, char diag, int n, const std::complex<float> *ap,
           std::complex<float> *x, int incx);
/** @copydoc stpsv */
void ztpsv(char uplo, char trans, char diag, int n, const std::complex<double> *ap,
           std::complex<double> *x, int incx);

/* ------------------------------------------------------------------ */
/* spr — Symmetric Packed Rank-1 Update                               */
/*   a ← α·x·xᵀ + a   (a symmetric, packed storage)                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision symmetric packed rank-1 update: a ← α·x·xᵀ + a.
 * @param uplo 'U' upper triangle packed, 'L' lower triangle packed.
 * @param n Order of the symmetric matrix a.
 * @param alpha Scalar multiplier.
 * @param x Input vector of length n.
 * @param incx Stride between elements of x.
 * @param ap Input/output packed storage array of n(n+1)/2 elements.
 */
void sspr(char uplo, int n, float alpha, const float *x, int incx, float *ap);
/** @copydoc sspr */
void dspr(char uplo, int n, double alpha, const double *x, int incx, double *ap);

/* ------------------------------------------------------------------ */
/* hpr — Hermitian Packed Rank-1 Update                               */
/*   a ← α·x·xᴴ + a   (a Hermitian, packed, α real)                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Complex-float Hermitian packed rank-1 update: a ← α·x·xᴴ + a.
 * @param uplo 'U' upper triangle packed, 'L' lower triangle packed.
 * @param n Order of the Hermitian matrix a.
 * @param alpha Real scalar multiplier.
 * @param x Input vector of length n.
 * @param incx Stride between elements of x.
 * @param ap Input/output packed storage array of n(n+1)/2 elements.
 */
void chpr(char uplo, int n, float alpha, const std::complex<float> *x, int incx,
          std::complex<float> *ap);
/** @copydoc chpr */
void zhpr(char uplo, int n, double alpha, const std::complex<double> *x, int incx,
          std::complex<double> *ap);

/* ------------------------------------------------------------------ */
/* spr2 — Symmetric Packed Rank-2 Update                              */
/*   a ← α·x·yᵀ + α·y·xᵀ + a   (a symmetric, packed)               */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision symmetric packed rank-2 update: a ← α·x·yᵀ + α·y·xᵀ + a.
 * @param uplo 'U' upper triangle packed, 'L' lower triangle packed.
 * @param n Order of the symmetric matrix a.
 * @param alpha Scalar multiplier.
 * @param x Input vector of length n.
 * @param incx Stride between elements of x.
 * @param y Input vector of length n.
 * @param incy Stride between elements of y.
 * @param ap Input/output packed storage array of n(n+1)/2 elements.
 */
void sspr2(char uplo, int n, float alpha, const float *x, int incx, const float *y, int incy,
           float *ap);
/** @copydoc sspr2 */
void dspr2(char uplo, int n, double alpha, const double *x, int incx, const double *y, int incy,
           double *ap);

/* ------------------------------------------------------------------ */
/* hpr2 — Hermitian Packed Rank-2 Update                              */
/*   a ← α·x·yᴴ + conj(α)·y·xᴴ + a   (a Hermitian, packed)         */
/* ------------------------------------------------------------------ */

/**
 * @brief Complex-float Hermitian packed rank-2 update: a ← α·x·yᴴ + conj(α)·y·xᴴ + a.
 * @param uplo 'U' upper triangle packed, 'L' lower triangle packed.
 * @param n Order of the Hermitian matrix a.
 * @param alpha Complex scalar multiplier.
 * @param x Input vector of length n.
 * @param incx Stride between elements of x.
 * @param y Input vector of length n.
 * @param incy Stride between elements of y.
 * @param ap Input/output packed storage array of n(n+1)/2 elements.
 */
void chpr2(char uplo, int n, std::complex<float> alpha, const std::complex<float> *x, int incx,
           const std::complex<float> *y, int incy, std::complex<float> *ap);
/** @copydoc chpr2 */
void zhpr2(char uplo, int n, std::complex<double> alpha, const std::complex<double> *x, int incx,
           const std::complex<double> *y, int incy, std::complex<double> *ap);

/** @} */

/** @defgroup level3_ops Level-3 Matrix-Matrix Operations
 *  @brief BLAS-like matrix-matrix operations using column-major storage.
 *
 *  Matrix element \f$a(i,j)\f$ is stored at offset \f$i+j\cdot lda\f$.
 *  Leading dimensions must be at least the number of stored rows (and at least one).
 *  `trans` accepts 'N', 'T', or 'C' for matrix multiply and triangular operations;
 *  symmetric rank updates accept 'N' or 'T', while Hermitian rank updates accept 'N'
 *  or 'C'. `uplo` accepts 'U' or 'L'; `diag` accepts 'U' or 'N'; `side` accepts 'L'
 *  or 'R'. Rank updates modify only the triangle selected by `uplo`. Invalid arguments
 *  are reported to the configured error handler using Netlib parameter numbering.
 *
 *  A zero row or column dimension is a quick return. Rank-k routines with `k == 0`
 *  apply `beta` to the selected output triangle. Triangular matrix operations honor
 *  the unit-diagonal flag without reading the stored diagonal.
 *  @{
 */

/**
 * @brief General matrix multiply: C ← alpha·op(A)·op(B) + beta·C.
 * @param transa Operation applied to A.
 * @param transb Operation applied to B.
 * @param m Rows of op(A) and C.
 * @param n Columns of op(B) and C.
 * @param k Columns of op(A) and rows of op(B).
 * @param alpha Product multiplier.
 * @param a A matrix in column-major storage.
 * @param lda Leading dimension of A.
 * @param b B matrix in column-major storage.
 * @param ldb Leading dimension of B.
 * @param beta Existing C multiplier.
 * @param c Input/output matrix C.
 * @param ldc Leading dimension of C.
 */
void sgemm(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda,
           const float *b, int ldb, float beta, float *c, int ldc);
/** @copydoc sgemm */
void dgemm(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda,
           const double *b, int ldb, double beta, double *c, int ldc);
/** @copydoc sgemm */
void cgemm(char transa, char transb, int m, int n, int k, std::complex<float> alpha,
           const std::complex<float> *a, int lda, const std::complex<float> *b, int ldb,
           std::complex<float> beta, std::complex<float> *c, int ldc);
/** @copydoc sgemm */
void zgemm(char transa, char transb, int m, int n, int k, std::complex<double> alpha,
           const std::complex<double> *a, int lda, const std::complex<double> *b, int ldb,
           std::complex<double> beta, std::complex<double> *c, int ldc);

/**
 * @brief Symmetric matrix multiply: C ← alpha·A·B + beta·C or C ← alpha·B·A + beta·C.
 * @param side 'L' for A·B, 'R' for B·A.
 * @param uplo Stored triangle of the symmetric matrix A.
 * @param m Rows of B and C.
 * @param n Columns of B and C.
 * @param alpha Product multiplier.
 * @param a Symmetric matrix A.
 * @param lda Leading dimension of A.
 * @param b Input matrix B.
 * @param ldb Leading dimension of B.
 * @param beta Existing C multiplier.
 * @param c Input/output matrix C.
 * @param ldc Leading dimension of C.
 */
void ssymm(char side, char uplo, int m, int n, float alpha, const float *a, int lda, const float *b,
           int ldb, float beta, float *c, int ldc);
/** @copydoc ssymm */
void dsymm(char side, char uplo, int m, int n, double alpha, const double *a, int lda,
           const double *b, int ldb, double beta, double *c, int ldc);
/**
 * @brief Hermitian matrix multiply: C ← alpha·A·B + beta·C or C ← alpha·B·A + beta·C.
 * @param side 'L' for A·B, 'R' for B·A.
 * @param uplo Stored triangle of the Hermitian matrix A.
 * @param m Rows of B and C.
 * @param n Columns of B and C.
 * @param alpha Product multiplier.
 * @param a Hermitian matrix A.
 * @param lda Leading dimension of A.
 * @param b Input matrix B.
 * @param ldb Leading dimension of B.
 * @param beta Existing C multiplier.
 * @param c Input/output matrix C.
 * @param ldc Leading dimension of C.
 */
void chemm(char side, char uplo, int m, int n, std::complex<float> alpha,
           const std::complex<float> *a, int lda, const std::complex<float> *b, int ldb,
           std::complex<float> beta, std::complex<float> *c, int ldc);
/** @copydoc chemm */
void zhemm(char side, char uplo, int m, int n, std::complex<double> alpha,
           const std::complex<double> *a, int lda, const std::complex<double> *b, int ldb,
           std::complex<double> beta, std::complex<double> *c, int ldc);

/**
 * @brief Symmetric rank-k update: C ← alpha·op(A)·op(A)ᵀ + beta·C.
 * @param uplo Triangle of C to update.
 * @param trans Operation applied to A ('N' or 'T').
 * @param n Order of C.
 * @param k Rank of the update.
 * @param alpha Product multiplier.
 * @param a Input matrix A.
 * @param lda Leading dimension of A.
 * @param beta Existing C multiplier.
 * @param c Input/output symmetric matrix C.
 * @param ldc Leading dimension of C.
 */
void ssyrk(char uplo, char trans, int n, int k, float alpha, const float *a, int lda, float beta,
           float *c, int ldc);
/** @copydoc ssyrk */
void dsyrk(char uplo, char trans, int n, int k, double alpha, const double *a, int lda, double beta,
           double *c, int ldc);
/**
 * @brief Hermitian rank-k update: C ← alpha·op(A)·op(A)ᴴ + beta·C.
 * @param uplo Triangle of C to update.
 * @param trans Operation applied to A ('N' or 'C').
 * @param n Order of C.
 * @param k Rank of the update.
 * @param alpha Real product multiplier.
 * @param a Input matrix A.
 * @param lda Leading dimension of A.
 * @param beta Real existing C multiplier.
 * @param c Input/output Hermitian matrix C.
 * @param ldc Leading dimension of C.
 */
void cherk(char uplo, char trans, int n, int k, float alpha, const std::complex<float> *a, int lda,
           float beta, std::complex<float> *c, int ldc);
/** @copydoc cherk */
void zherk(char uplo, char trans, int n, int k, double alpha, const std::complex<double> *a,
           int lda, double beta, std::complex<double> *c, int ldc);

/**
 * @brief Symmetric rank-2k update: C ← alpha·op(A)·op(B)ᵀ + alpha·op(B)·op(A)ᵀ + beta·C.
 * @param uplo Triangle of C to update.
 * @param trans Operation applied to A and B ('N' or 'T').
 * @param n Order of C.
 * @param k Rank of the update.
 * @param alpha Product multiplier.
 * @param a First input matrix.
 * @param lda Leading dimension of A.
 * @param b Second input matrix.
 * @param ldb Leading dimension of B.
 * @param beta Existing C multiplier.
 * @param c Input/output symmetric matrix C.
 * @param ldc Leading dimension of C.
 */
void ssyr2k(char uplo, char trans, int n, int k, float alpha, const float *a, int lda,
            const float *b, int ldb, float beta, float *c, int ldc);
/** @copydoc ssyr2k */
void dsyr2k(char uplo, char trans, int n, int k, double alpha, const double *a, int lda,
            const double *b, int ldb, double beta, double *c, int ldc);
/**
 * @brief Hermitian rank-2k update with complex alpha and real beta.
 * @param uplo Triangle of C to update.
 * @param trans Operation applied to A and B ('N' or 'C').
 * @param n Order of C.
 * @param k Rank of the update.
 * @param alpha Complex product multiplier.
 * @param a First input matrix.
 * @param lda Leading dimension of A.
 * @param b Second input matrix.
 * @param ldb Leading dimension of B.
 * @param beta Real existing C multiplier.
 * @param c Input/output Hermitian matrix C.
 * @param ldc Leading dimension of C.
 */
void cher2k(char uplo, char trans, int n, int k, std::complex<float> alpha,
            const std::complex<float> *a, int lda, const std::complex<float> *b, int ldb,
            float beta, std::complex<float> *c, int ldc);
/** @copydoc cher2k */
void zher2k(char uplo, char trans, int n, int k, std::complex<double> alpha,
            const std::complex<double> *a, int lda, const std::complex<double> *b, int ldb,
            double beta, std::complex<double> *c, int ldc);

/**
 * @brief Triangular matrix multiply: B ← alpha·op(A)·B or B ← alpha·B·op(A).
 * @param side 'L' for op(A)·B, 'R' for B·op(A).
 * @param uplo Stored triangle of A.
 * @param trans Operation applied to A.
 * @param diag 'U' for unit diagonal, 'N' for non-unit diagonal.
 * @param m Rows of B.
 * @param n Columns of B.
 * @param alpha Scalar multiplier.
 * @param a Triangular matrix A.
 * @param lda Leading dimension of A.
 * @param b Input/output matrix B.
 * @param ldb Leading dimension of B.
 */
void strmm(char side, char uplo, char trans, char diag, int m, int n, float alpha, const float *a,
           int lda, float *b, int ldb);
/** @copydoc strmm */
void dtrmm(char side, char uplo, char trans, char diag, int m, int n, double alpha, const double *a,
           int lda, double *b, int ldb);
/** @copydoc strmm */
void ctrmm(char side, char uplo, char trans, char diag, int m, int n, std::complex<float> alpha,
           const std::complex<float> *a, int lda, std::complex<float> *b, int ldb);
/** @copydoc strmm */
void ztrmm(char side, char uplo, char trans, char diag, int m, int n, std::complex<double> alpha,
           const std::complex<double> *a, int lda, std::complex<double> *b, int ldb);

/**
 * @brief Triangular solve: op(A)·X = alpha·B or X·op(A) = alpha·B; X overwrites B.
 * @param side 'L' solves op(A)·X = alpha·B; 'R' solves X·op(A) = alpha·B.
 * @param uplo Stored triangle of A.
 * @param trans Operation applied to A.
 * @param diag 'U' for unit diagonal, 'N' for non-unit diagonal.
 * @param m Rows of B.
 * @param n Columns of B.
 * @param alpha Right-hand-side multiplier.
 * @param a Triangular matrix A.
 * @param lda Leading dimension of A.
 * @param b On entry, the right-hand side; on exit, the solution.
 * @param ldb Leading dimension of B.
 */
void strsm(char side, char uplo, char trans, char diag, int m, int n, float alpha, const float *a,
           int lda, float *b, int ldb);
/** @copydoc strsm */
void dtrsm(char side, char uplo, char trans, char diag, int m, int n, double alpha, const double *a,
           int lda, double *b, int ldb);
/** @copydoc strsm */
void ctrsm(char side, char uplo, char trans, char diag, int m, int n, std::complex<float> alpha,
           const std::complex<float> *a, int lda, std::complex<float> *b, int ldb);
/** @copydoc strsm */
void ztrsm(char side, char uplo, char trans, char diag, int m, int n, std::complex<double> alpha,
           const std::complex<double> *a, int lda, std::complex<double> *b, int ldb);

/** @} */

} // namespace theblas
