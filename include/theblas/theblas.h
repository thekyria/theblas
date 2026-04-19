#pragma once

#include "theblas/version.h"

#include <complex>
#include <cstddef>

namespace theblas {

/**
 * @file theblas.h
 * @brief Public C++17 API for a minimal Level-1 BLAS-like library.
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
 *  Level-2 routines operate on a matrix **A** and one or two vectors.
 *  Matrices are stored in **column-major** order (Fortran layout):
 *  element \f$A(i,j)\f$ resides at offset \f$i + j \cdot \text{lda}\f$.
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
/*   y ← α·op(A)·x + β·y                                            */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision general matrix-vector multiply: y ← α·op(A)·x + β·y.
 * @param trans 'N' no-transpose, 'T' transpose, 'C' conjugate-transpose.
 * @param m Number of rows of A.
 * @param n Number of columns of A.
 * @param alpha Scalar multiplier for op(A)·x.
 * @param A Pointer to the m×n matrix in column-major order.
 * @param lda Leading dimension of A (≥ max(1,m)).
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @param beta Scalar multiplier for y.
 * @param y Input/output vector.
 * @param incy Stride between elements of y.
 */
void sgemv(char trans, int m, int n, float alpha, const float *A, int lda, const float *x,
           int incx, float beta, float *y, int incy);
/** @copydoc sgemv */
void dgemv(char trans, int m, int n, double alpha, const double *A, int lda, const double *x,
           int incx, double beta, double *y, int incy);
/** @copydoc sgemv */
void cgemv(char trans, int m, int n, std::complex<float> alpha, const std::complex<float> *A,
           int lda, const std::complex<float> *x, int incx, std::complex<float> beta,
           std::complex<float> *y, int incy);
/** @copydoc sgemv */
void zgemv(char trans, int m, int n, std::complex<double> alpha, const std::complex<double> *A,
           int lda, const std::complex<double> *x, int incx, std::complex<double> beta,
           std::complex<double> *y, int incy);

/* ------------------------------------------------------------------ */
/* symv — Symmetric Matrix-Vector Multiply                            */
/*   y ← α·A·x + β·y   (A symmetric)                                */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision symmetric matrix-vector multiply: y ← α·A·x + β·y.
 * @param uplo 'U' upper triangle stored, 'L' lower triangle stored.
 * @param n Order of the symmetric matrix A.
 * @param alpha Scalar multiplier for A·x.
 * @param A Pointer to the n×n symmetric matrix in column-major order.
 * @param lda Leading dimension of A (≥ max(1,n)).
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @param beta Scalar multiplier for y.
 * @param y Input/output vector.
 * @param incy Stride between elements of y.
 */
void ssymv(char uplo, int n, float alpha, const float *A, int lda, const float *x, int incx,
           float beta, float *y, int incy);
/** @copydoc ssymv */
void dsymv(char uplo, int n, double alpha, const double *A, int lda, const double *x, int incx,
           double beta, double *y, int incy);

/* ------------------------------------------------------------------ */
/* hemv — Hermitian Matrix-Vector Multiply                            */
/*   y ← α·A·x + β·y   (A Hermitian)                                */
/* ------------------------------------------------------------------ */

/**
 * @brief Complex-float Hermitian matrix-vector multiply: y ← α·A·x + β·y.
 * @param uplo 'U' upper triangle stored, 'L' lower triangle stored.
 * @param n Order of the Hermitian matrix A.
 * @param alpha Scalar multiplier for A·x.
 * @param A Pointer to the n×n Hermitian matrix in column-major order.
 * @param lda Leading dimension of A (≥ max(1,n)).
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @param beta Scalar multiplier for y.
 * @param y Input/output vector.
 * @param incy Stride between elements of y.
 */
void chemv(char uplo, int n, std::complex<float> alpha, const std::complex<float> *A, int lda,
           const std::complex<float> *x, int incx, std::complex<float> beta,
           std::complex<float> *y, int incy);
/** @copydoc chemv */
void zhemv(char uplo, int n, std::complex<double> alpha, const std::complex<double> *A, int lda,
           const std::complex<double> *x, int incx, std::complex<double> beta,
           std::complex<double> *y, int incy);

/* ------------------------------------------------------------------ */
/* trmv — Triangular Matrix-Vector Multiply                           */
/*   x ← op(A)·x   (A triangular)                                   */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision triangular matrix-vector multiply: x ← op(A)·x.
 * @param uplo 'U' upper triangular, 'L' lower triangular.
 * @param trans 'N' no-transpose, 'T' transpose, 'C' conjugate-transpose.
 * @param diag 'U' unit diagonal, 'N' non-unit diagonal.
 * @param n Order of the triangular matrix A.
 * @param A Pointer to the n×n triangular matrix in column-major order.
 * @param lda Leading dimension of A (≥ max(1,n)).
 * @param x Input/output vector.
 * @param incx Stride between elements of x.
 */
void strmv(char uplo, char trans, char diag, int n, const float *A, int lda, float *x, int incx);
/** @copydoc strmv */
void dtrmv(char uplo, char trans, char diag, int n, const double *A, int lda, double *x,
           int incx);
/** @copydoc strmv */
void ctrmv(char uplo, char trans, char diag, int n, const std::complex<float> *A, int lda,
           std::complex<float> *x, int incx);
/** @copydoc strmv */
void ztrmv(char uplo, char trans, char diag, int n, const std::complex<double> *A, int lda,
           std::complex<double> *x, int incx);

/* ------------------------------------------------------------------ */
/* trsv — Triangular Solve                                            */
/*   solve op(A)·x = b   (A triangular, x overwrites b)              */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision triangular solve: solve op(A)·x = b.
 * @param uplo 'U' upper triangular, 'L' lower triangular.
 * @param trans 'N' no-transpose, 'T' transpose, 'C' conjugate-transpose.
 * @param diag 'U' unit diagonal, 'N' non-unit diagonal.
 * @param n Order of the triangular matrix A.
 * @param A Pointer to the n×n triangular matrix in column-major order.
 * @param lda Leading dimension of A (≥ max(1,n)).
 * @param x On entry, the right-hand side b; on exit, the solution x.
 * @param incx Stride between elements of x.
 */
void strsv(char uplo, char trans, char diag, int n, const float *A, int lda, float *x, int incx);
/** @copydoc strsv */
void dtrsv(char uplo, char trans, char diag, int n, const double *A, int lda, double *x,
           int incx);
/** @copydoc strsv */
void ctrsv(char uplo, char trans, char diag, int n, const std::complex<float> *A, int lda,
           std::complex<float> *x, int incx);
/** @copydoc strsv */
void ztrsv(char uplo, char trans, char diag, int n, const std::complex<double> *A, int lda,
           std::complex<double> *x, int incx);

/* ------------------------------------------------------------------ */
/* ger — General Rank-1 Update (real)                                 */
/*   A ← α·x·yᵀ + A                                                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision rank-1 update: A ← α·x·yᵀ + A.
 * @param m Number of rows of A.
 * @param n Number of columns of A.
 * @param alpha Scalar multiplier.
 * @param x Input vector of length m.
 * @param incx Stride between elements of x.
 * @param y Input vector of length n.
 * @param incy Stride between elements of y.
 * @param A Input/output m×n matrix in column-major order.
 * @param lda Leading dimension of A (≥ max(1,m)).
 */
void sger(int m, int n, float alpha, const float *x, int incx, const float *y, int incy,
          float *A, int lda);
/** @copydoc sger */
void dger(int m, int n, double alpha, const double *x, int incx, const double *y, int incy,
          double *A, int lda);

/* ------------------------------------------------------------------ */
/* geru / gerc — General Rank-1 Update (complex)                      */
/*   geru: A ← α·x·yᵀ + A   (unconjugated)                          */
/*   gerc: A ← α·x·yᴴ + A   (conjugated)                            */
/* ------------------------------------------------------------------ */

/**
 * @brief Complex-float unconjugated rank-1 update: A ← α·x·yᵀ + A.
 * @param m Number of rows of A.
 * @param n Number of columns of A.
 * @param alpha Scalar multiplier.
 * @param x Input vector of length m.
 * @param incx Stride between elements of x.
 * @param y Input vector of length n.
 * @param incy Stride between elements of y.
 * @param A Input/output m×n matrix in column-major order.
 * @param lda Leading dimension of A (≥ max(1,m)).
 */
void cgeru(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int incx,
           const std::complex<float> *y, int incy, std::complex<float> *A, int lda);
/** @copydoc cgeru */
void zgeru(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int incx,
           const std::complex<double> *y, int incy, std::complex<double> *A, int lda);

/**
 * @brief Complex-float conjugated rank-1 update: A ← α·x·yᴴ + A.
 * @param m Number of rows of A.
 * @param n Number of columns of A.
 * @param alpha Scalar multiplier.
 * @param x Input vector of length m.
 * @param incx Stride between elements of x.
 * @param y Input vector of length n (conjugated in the product).
 * @param incy Stride between elements of y.
 * @param A Input/output m×n matrix in column-major order.
 * @param lda Leading dimension of A (≥ max(1,m)).
 */
void cgerc(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int incx,
           const std::complex<float> *y, int incy, std::complex<float> *A, int lda);
/** @copydoc cgerc */
void zgerc(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int incx,
           const std::complex<double> *y, int incy, std::complex<double> *A, int lda);

/* ------------------------------------------------------------------ */
/* syr — Symmetric Rank-1 Update                                      */
/*   A ← α·x·xᵀ + A   (A symmetric)                                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision symmetric rank-1 update: A ← α·x·xᵀ + A.
 * @param uplo 'U' upper triangle stored, 'L' lower triangle stored.
 * @param n Order of the symmetric matrix A.
 * @param alpha Scalar multiplier.
 * @param x Input vector of length n.
 * @param incx Stride between elements of x.
 * @param A Input/output n×n symmetric matrix in column-major order.
 * @param lda Leading dimension of A (≥ max(1,n)).
 */
void ssyr(char uplo, int n, float alpha, const float *x, int incx, float *A, int lda);
/** @copydoc ssyr */
void dsyr(char uplo, int n, double alpha, const double *x, int incx, double *A, int lda);

/* ------------------------------------------------------------------ */
/* her — Hermitian Rank-1 Update                                      */
/*   A ← α·x·xᴴ + A   (A Hermitian, α real)                         */
/* ------------------------------------------------------------------ */

/**
 * @brief Complex-float Hermitian rank-1 update: A ← α·x·xᴴ + A.
 * @param uplo 'U' upper triangle stored, 'L' lower triangle stored.
 * @param n Order of the Hermitian matrix A.
 * @param alpha Real scalar multiplier.
 * @param x Input vector of length n.
 * @param incx Stride between elements of x.
 * @param A Input/output n×n Hermitian matrix in column-major order.
 * @param lda Leading dimension of A (≥ max(1,n)).
 */
void cher(char uplo, int n, float alpha, const std::complex<float> *x, int incx,
          std::complex<float> *A, int lda);
/** @copydoc cher */
void zher(char uplo, int n, double alpha, const std::complex<double> *x, int incx,
          std::complex<double> *A, int lda);

/* ------------------------------------------------------------------ */
/* syr2 — Symmetric Rank-2 Update                                     */
/*   A ← α·x·yᵀ + α·y·xᵀ + A   (A symmetric)                       */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision symmetric rank-2 update: A ← α·x·yᵀ + α·y·xᵀ + A.
 * @param uplo 'U' upper triangle stored, 'L' lower triangle stored.
 * @param n Order of the symmetric matrix A.
 * @param alpha Scalar multiplier.
 * @param x Input vector of length n.
 * @param incx Stride between elements of x.
 * @param y Input vector of length n.
 * @param incy Stride between elements of y.
 * @param A Input/output n×n symmetric matrix in column-major order.
 * @param lda Leading dimension of A (≥ max(1,n)).
 */
void ssyr2(char uplo, int n, float alpha, const float *x, int incx, const float *y, int incy,
           float *A, int lda);
/** @copydoc ssyr2 */
void dsyr2(char uplo, int n, double alpha, const double *x, int incx, const double *y, int incy,
           double *A, int lda);

/* ------------------------------------------------------------------ */
/* her2 — Hermitian Rank-2 Update                                     */
/*   A ← α·x·yᴴ + conj(α)·y·xᴴ + A   (A Hermitian)                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Complex-float Hermitian rank-2 update: A ← α·x·yᴴ + conj(α)·y·xᴴ + A.
 * @param uplo 'U' upper triangle stored, 'L' lower triangle stored.
 * @param n Order of the Hermitian matrix A.
 * @param alpha Complex scalar multiplier.
 * @param x Input vector of length n.
 * @param incx Stride between elements of x.
 * @param y Input vector of length n.
 * @param incy Stride between elements of y.
 * @param A Input/output n×n Hermitian matrix in column-major order.
 * @param lda Leading dimension of A (≥ max(1,n)).
 */
void cher2(char uplo, int n, std::complex<float> alpha, const std::complex<float> *x, int incx,
           const std::complex<float> *y, int incy, std::complex<float> *A, int lda);
/** @copydoc cher2 */
void zher2(char uplo, int n, std::complex<double> alpha, const std::complex<double> *x, int incx,
           const std::complex<double> *y, int incy, std::complex<double> *A, int lda);

/* ------------------------------------------------------------------ */
/* gbmv — General Band Matrix-Vector Multiply                         */
/*   y ← α·op(A)·x + β·y   (A banded)                               */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision general band matrix-vector multiply: y ← α·op(A)·x + β·y.
 * @param trans 'N' no-transpose, 'T' transpose, 'C' conjugate-transpose.
 * @param m Number of rows of A.
 * @param n Number of columns of A.
 * @param kl Number of sub-diagonals.
 * @param ku Number of super-diagonals.
 * @param alpha Scalar multiplier for op(A)·x.
 * @param A Pointer to the band matrix in column-major band storage.
 * @param lda Leading dimension of A (≥ kl+ku+1).
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @param beta Scalar multiplier for y.
 * @param y Input/output vector.
 * @param incy Stride between elements of y.
 */
void sgbmv(char trans, int m, int n, int kl, int ku, float alpha, const float *A, int lda,
           const float *x, int incx, float beta, float *y, int incy);
/** @copydoc sgbmv */
void dgbmv(char trans, int m, int n, int kl, int ku, double alpha, const double *A, int lda,
           const double *x, int incx, double beta, double *y, int incy);
/** @copydoc sgbmv */
void cgbmv(char trans, int m, int n, int kl, int ku, std::complex<float> alpha,
           const std::complex<float> *A, int lda, const std::complex<float> *x, int incx,
           std::complex<float> beta, std::complex<float> *y, int incy);
/** @copydoc sgbmv */
void zgbmv(char trans, int m, int n, int kl, int ku, std::complex<double> alpha,
           const std::complex<double> *A, int lda, const std::complex<double> *x, int incx,
           std::complex<double> beta, std::complex<double> *y, int incy);

/* ------------------------------------------------------------------ */
/* sbmv — Symmetric Band Matrix-Vector Multiply                       */
/*   y ← α·A·x + β·y   (A symmetric banded)                         */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision symmetric band matrix-vector multiply: y ← α·A·x + β·y.
 * @param uplo 'U' upper triangle stored, 'L' lower triangle stored.
 * @param n Order of the symmetric matrix A.
 * @param k Number of super-diagonals (bandwidth).
 * @param alpha Scalar multiplier for A·x.
 * @param A Pointer to the symmetric band matrix in column-major band storage.
 * @param lda Leading dimension of A (≥ k+1).
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @param beta Scalar multiplier for y.
 * @param y Input/output vector.
 * @param incy Stride between elements of y.
 */
void ssbmv(char uplo, int n, int k, float alpha, const float *A, int lda, const float *x,
           int incx, float beta, float *y, int incy);
/** @copydoc ssbmv */
void dsbmv(char uplo, int n, int k, double alpha, const double *A, int lda, const double *x,
           int incx, double beta, double *y, int incy);

/* ------------------------------------------------------------------ */
/* hbmv — Hermitian Band Matrix-Vector Multiply                       */
/*   y ← α·A·x + β·y   (A Hermitian banded)                         */
/* ------------------------------------------------------------------ */

/**
 * @brief Complex-float Hermitian band matrix-vector multiply: y ← α·A·x + β·y.
 * @param uplo 'U' upper triangle stored, 'L' lower triangle stored.
 * @param n Order of the Hermitian matrix A.
 * @param k Number of super-diagonals (bandwidth).
 * @param alpha Scalar multiplier for A·x.
 * @param A Pointer to the Hermitian band matrix in column-major band storage.
 * @param lda Leading dimension of A (≥ k+1).
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @param beta Scalar multiplier for y.
 * @param y Input/output vector.
 * @param incy Stride between elements of y.
 */
void chbmv(char uplo, int n, int k, std::complex<float> alpha, const std::complex<float> *A,
           int lda, const std::complex<float> *x, int incx, std::complex<float> beta,
           std::complex<float> *y, int incy);
/** @copydoc chbmv */
void zhbmv(char uplo, int n, int k, std::complex<double> alpha, const std::complex<double> *A,
           int lda, const std::complex<double> *x, int incx, std::complex<double> beta,
           std::complex<double> *y, int incy);

/* ------------------------------------------------------------------ */
/* tbmv — Triangular Band Matrix-Vector Multiply                      */
/*   x ← op(A)·x   (A triangular banded)                             */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision triangular band matrix-vector multiply: x ← op(A)·x.
 * @param uplo 'U' upper triangular, 'L' lower triangular.
 * @param trans 'N' no-transpose, 'T' transpose, 'C' conjugate-transpose.
 * @param diag 'U' unit diagonal, 'N' non-unit diagonal.
 * @param n Order of the triangular matrix A.
 * @param k Number of super-diagonals (uplo='U') or sub-diagonals (uplo='L').
 * @param A Pointer to the triangular band matrix in column-major band storage.
 * @param lda Leading dimension of A (≥ k+1).
 * @param x Input/output vector.
 * @param incx Stride between elements of x.
 */
void stbmv(char uplo, char trans, char diag, int n, int k, const float *A, int lda, float *x,
           int incx);
/** @copydoc stbmv */
void dtbmv(char uplo, char trans, char diag, int n, int k, const double *A, int lda, double *x,
           int incx);
/** @copydoc stbmv */
void ctbmv(char uplo, char trans, char diag, int n, int k, const std::complex<float> *A, int lda,
           std::complex<float> *x, int incx);
/** @copydoc stbmv */
void ztbmv(char uplo, char trans, char diag, int n, int k, const std::complex<double> *A,
           int lda, std::complex<double> *x, int incx);

/* ------------------------------------------------------------------ */
/* tbsv — Triangular Band Solve                                       */
/*   solve op(A)·x = b   (A triangular banded)                       */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision triangular band solve: solve op(A)·x = b.
 * @param uplo 'U' upper triangular, 'L' lower triangular.
 * @param trans 'N' no-transpose, 'T' transpose, 'C' conjugate-transpose.
 * @param diag 'U' unit diagonal, 'N' non-unit diagonal.
 * @param n Order of the triangular matrix A.
 * @param k Number of super-diagonals (uplo='U') or sub-diagonals (uplo='L').
 * @param A Pointer to the triangular band matrix in column-major band storage.
 * @param lda Leading dimension of A (≥ k+1).
 * @param x On entry, the right-hand side b; on exit, the solution x.
 * @param incx Stride between elements of x.
 */
void stbsv(char uplo, char trans, char diag, int n, int k, const float *A, int lda, float *x,
           int incx);
/** @copydoc stbsv */
void dtbsv(char uplo, char trans, char diag, int n, int k, const double *A, int lda, double *x,
           int incx);
/** @copydoc stbsv */
void ctbsv(char uplo, char trans, char diag, int n, int k, const std::complex<float> *A, int lda,
           std::complex<float> *x, int incx);
/** @copydoc stbsv */
void ztbsv(char uplo, char trans, char diag, int n, int k, const std::complex<double> *A,
           int lda, std::complex<double> *x, int incx);

/* ------------------------------------------------------------------ */
/* spmv — Symmetric Packed Matrix-Vector Multiply                     */
/*   y ← α·A·x + β·y   (A symmetric, packed storage)                */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision symmetric packed matrix-vector multiply: y ← α·A·x + β·y.
 * @param uplo 'U' upper triangle packed, 'L' lower triangle packed.
 * @param n Order of the symmetric matrix A.
 * @param alpha Scalar multiplier for A·x.
 * @param Ap Packed storage array of n(n+1)/2 elements.
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @param beta Scalar multiplier for y.
 * @param y Input/output vector.
 * @param incy Stride between elements of y.
 */
void sspmv(char uplo, int n, float alpha, const float *Ap, const float *x, int incx, float beta,
           float *y, int incy);
/** @copydoc sspmv */
void dspmv(char uplo, int n, double alpha, const double *Ap, const double *x, int incx,
           double beta, double *y, int incy);

/* ------------------------------------------------------------------ */
/* hpmv — Hermitian Packed Matrix-Vector Multiply                     */
/*   y ← α·A·x + β·y   (A Hermitian, packed storage)                */
/* ------------------------------------------------------------------ */

/**
 * @brief Complex-float Hermitian packed matrix-vector multiply: y ← α·A·x + β·y.
 * @param uplo 'U' upper triangle packed, 'L' lower triangle packed.
 * @param n Order of the Hermitian matrix A.
 * @param alpha Scalar multiplier for A·x.
 * @param Ap Packed storage array of n(n+1)/2 elements.
 * @param x Input vector.
 * @param incx Stride between elements of x.
 * @param beta Scalar multiplier for y.
 * @param y Input/output vector.
 * @param incy Stride between elements of y.
 */
void chpmv(char uplo, int n, std::complex<float> alpha, const std::complex<float> *Ap,
           const std::complex<float> *x, int incx, std::complex<float> beta,
           std::complex<float> *y, int incy);
/** @copydoc chpmv */
void zhpmv(char uplo, int n, std::complex<double> alpha, const std::complex<double> *Ap,
           const std::complex<double> *x, int incx, std::complex<double> beta,
           std::complex<double> *y, int incy);

/* ------------------------------------------------------------------ */
/* tpmv — Triangular Packed Matrix-Vector Multiply                    */
/*   x ← op(A)·x   (A triangular, packed storage)                   */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision triangular packed matrix-vector multiply: x ← op(A)·x.
 * @param uplo 'U' upper triangular packed, 'L' lower triangular packed.
 * @param trans 'N' no-transpose, 'T' transpose, 'C' conjugate-transpose.
 * @param diag 'U' unit diagonal, 'N' non-unit diagonal.
 * @param n Order of the triangular matrix A.
 * @param Ap Packed storage array of n(n+1)/2 elements.
 * @param x Input/output vector.
 * @param incx Stride between elements of x.
 */
void stpmv(char uplo, char trans, char diag, int n, const float *Ap, float *x, int incx);
/** @copydoc stpmv */
void dtpmv(char uplo, char trans, char diag, int n, const double *Ap, double *x, int incx);
/** @copydoc stpmv */
void ctpmv(char uplo, char trans, char diag, int n, const std::complex<float> *Ap,
           std::complex<float> *x, int incx);
/** @copydoc stpmv */
void ztpmv(char uplo, char trans, char diag, int n, const std::complex<double> *Ap,
           std::complex<double> *x, int incx);

/* ------------------------------------------------------------------ */
/* tpsv — Triangular Packed Solve                                     */
/*   solve op(A)·x = b   (A triangular, packed storage)              */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision triangular packed solve: solve op(A)·x = b.
 * @param uplo 'U' upper triangular packed, 'L' lower triangular packed.
 * @param trans 'N' no-transpose, 'T' transpose, 'C' conjugate-transpose.
 * @param diag 'U' unit diagonal, 'N' non-unit diagonal.
 * @param n Order of the triangular matrix A.
 * @param Ap Packed storage array of n(n+1)/2 elements.
 * @param x On entry, the right-hand side b; on exit, the solution x.
 * @param incx Stride between elements of x.
 */
void stpsv(char uplo, char trans, char diag, int n, const float *Ap, float *x, int incx);
/** @copydoc stpsv */
void dtpsv(char uplo, char trans, char diag, int n, const double *Ap, double *x, int incx);
/** @copydoc stpsv */
void ctpsv(char uplo, char trans, char diag, int n, const std::complex<float> *Ap,
           std::complex<float> *x, int incx);
/** @copydoc stpsv */
void ztpsv(char uplo, char trans, char diag, int n, const std::complex<double> *Ap,
           std::complex<double> *x, int incx);

/* ------------------------------------------------------------------ */
/* spr — Symmetric Packed Rank-1 Update                               */
/*   A ← α·x·xᵀ + A   (A symmetric, packed storage)                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision symmetric packed rank-1 update: A ← α·x·xᵀ + A.
 * @param uplo 'U' upper triangle packed, 'L' lower triangle packed.
 * @param n Order of the symmetric matrix A.
 * @param alpha Scalar multiplier.
 * @param x Input vector of length n.
 * @param incx Stride between elements of x.
 * @param Ap Input/output packed storage array of n(n+1)/2 elements.
 */
void sspr(char uplo, int n, float alpha, const float *x, int incx, float *Ap);
/** @copydoc sspr */
void dspr(char uplo, int n, double alpha, const double *x, int incx, double *Ap);

/* ------------------------------------------------------------------ */
/* hpr — Hermitian Packed Rank-1 Update                               */
/*   A ← α·x·xᴴ + A   (A Hermitian, packed, α real)                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Complex-float Hermitian packed rank-1 update: A ← α·x·xᴴ + A.
 * @param uplo 'U' upper triangle packed, 'L' lower triangle packed.
 * @param n Order of the Hermitian matrix A.
 * @param alpha Real scalar multiplier.
 * @param x Input vector of length n.
 * @param incx Stride between elements of x.
 * @param Ap Input/output packed storage array of n(n+1)/2 elements.
 */
void chpr(char uplo, int n, float alpha, const std::complex<float> *x, int incx,
          std::complex<float> *Ap);
/** @copydoc chpr */
void zhpr(char uplo, int n, double alpha, const std::complex<double> *x, int incx,
          std::complex<double> *Ap);

/* ------------------------------------------------------------------ */
/* spr2 — Symmetric Packed Rank-2 Update                              */
/*   A ← α·x·yᵀ + α·y·xᵀ + A   (A symmetric, packed)               */
/* ------------------------------------------------------------------ */

/**
 * @brief Single-precision symmetric packed rank-2 update: A ← α·x·yᵀ + α·y·xᵀ + A.
 * @param uplo 'U' upper triangle packed, 'L' lower triangle packed.
 * @param n Order of the symmetric matrix A.
 * @param alpha Scalar multiplier.
 * @param x Input vector of length n.
 * @param incx Stride between elements of x.
 * @param y Input vector of length n.
 * @param incy Stride between elements of y.
 * @param Ap Input/output packed storage array of n(n+1)/2 elements.
 */
void sspr2(char uplo, int n, float alpha, const float *x, int incx, const float *y, int incy,
           float *Ap);
/** @copydoc sspr2 */
void dspr2(char uplo, int n, double alpha, const double *x, int incx, const double *y, int incy,
           double *Ap);

/* ------------------------------------------------------------------ */
/* hpr2 — Hermitian Packed Rank-2 Update                              */
/*   A ← α·x·yᴴ + conj(α)·y·xᴴ + A   (A Hermitian, packed)         */
/* ------------------------------------------------------------------ */

/**
 * @brief Complex-float Hermitian packed rank-2 update: A ← α·x·yᴴ + conj(α)·y·xᴴ + A.
 * @param uplo 'U' upper triangle packed, 'L' lower triangle packed.
 * @param n Order of the Hermitian matrix A.
 * @param alpha Complex scalar multiplier.
 * @param x Input vector of length n.
 * @param incx Stride between elements of x.
 * @param y Input vector of length n.
 * @param incy Stride between elements of y.
 * @param Ap Input/output packed storage array of n(n+1)/2 elements.
 */
void chpr2(char uplo, int n, std::complex<float> alpha, const std::complex<float> *x, int incx,
           const std::complex<float> *y, int incy, std::complex<float> *Ap);
/** @copydoc chpr2 */
void zhpr2(char uplo, int n, std::complex<double> alpha, const std::complex<double> *x, int incx,
           const std::complex<double> *y, int incy, std::complex<double> *Ap);

/** @} */

} // namespace theblas
