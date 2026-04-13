#pragma once

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

} // namespace theblas
