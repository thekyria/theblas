#pragma once

#include <cstddef>
#include <complex>

namespace theblas {

// Level 1 BLAS (real) routines.
void sswap(int n, float* x, int incx, float* y, int incy);
void dswap(int n, double* x, int incx, double* y, int incy);

void cswap(int n, std::complex<float>* x, int incx, std::complex<float>* y, int incy);
void zswap(int n, std::complex<double>* x, int incx, std::complex<double>* y, int incy);

void scopy(int n, const float* x, int incx, float* y, int incy);
void dcopy(int n, const double* x, int incx, double* y, int incy);

void ccopy(int n, const std::complex<float>* x, int incx, std::complex<float>* y, int incy);
void zcopy(int n, const std::complex<double>* x, int incx, std::complex<double>* y, int incy);

void saxpy(int n, float alpha, const float* x, int incx, float* y, int incy);
void daxpy(int n, double alpha, const double* x, int incx, double* y, int incy);

void caxpy(int n, std::complex<float> alpha, const std::complex<float>* x, int incx, std::complex<float>* y, int incy);
void zaxpy(int n, std::complex<double> alpha, const std::complex<double>* x, int incx, std::complex<double>* y, int incy);

void sscal(int n, float alpha, float* x, int incx);
void dscal(int n, double alpha, double* x, int incx);

void cscal(int n, std::complex<float> alpha, std::complex<float>* x, int incx);
void zscal(int n, std::complex<double> alpha, std::complex<double>* x, int incx);
void csscal(int n, float alpha, std::complex<float>* x, int incx);
void zdscal(int n, double alpha, std::complex<double>* x, int incx);

float sdot(int n, const float* x, int incx, const float* y, int incy);
double ddot(int n, const double* x, int incx, const double* y, int incy);

std::complex<float> cdotu(int n, const std::complex<float>* x, int incx, const std::complex<float>* y, int incy);
std::complex<float> cdotc(int n, const std::complex<float>* x, int incx, const std::complex<float>* y, int incy);
std::complex<double> zdotu(int n, const std::complex<double>* x, int incx, const std::complex<double>* y, int incy);
std::complex<double> zdotc(int n, const std::complex<double>* x, int incx, const std::complex<double>* y, int incy);

float snrm2(int n, const float* x, int incx);
double dnrm2(int n, const double* x, int incx);

float scnrm2(int n, const std::complex<float>* x, int incx);
double dznrm2(int n, const std::complex<double>* x, int incx);

float sasum(int n, const float* x, int incx);
double dasum(int n, const double* x, int incx);

float scasum(int n, const std::complex<float>* x, int incx);
double dzasum(int n, const std::complex<double>* x, int incx);

void srot(int n, float* x, int incx, float* y, int incy, float c, float s);
void drot(int n, double* x, int incx, double* y, int incy, double c, double s);
void csrot(int n, std::complex<float>* x, int incx, std::complex<float>* y, int incy, float c, float s);
void zdrot(int n, std::complex<double>* x, int incx, std::complex<double>* y, int incy, double c, double s);

void srotg(float* a, float* b, float* c, float* s);
void drotg(double* a, double* b, double* c, double* s);
void crotg(std::complex<float>* a, std::complex<float> b, float* c, std::complex<float>* s);
void zrotg(std::complex<double>* a, std::complex<double> b, double* c, std::complex<double>* s);

void srotm(int n, float* x, int incx, float* y, int incy, const float* param);
void drotm(int n, double* x, int incx, double* y, int incy, const double* param);

void srotmg(float* d1, float* d2, float* b1, float b2, float* param);
void drotmg(double* d1, double* d2, double* b1, double b2, double* param);

// Netlib-style index of max absolute value, returned as 1-based index.
int isamax(int n, const float* x, int incx);
int idamax(int n, const double* x, int incx);
int icamax(int n, const std::complex<float>* x, int incx);
int izamax(int n, const std::complex<double>* x, int incx);

} // namespace theblas
  