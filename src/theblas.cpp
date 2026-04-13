#include "theblas/theblas.h"

#include <algorithm>
#include <cmath>
#include <complex>

namespace theblas {

namespace {

int start_index(int n, int inc) {
	return (inc > 0) ? 0 : (1 - n) * inc;
}

template <typename T>
void swap_impl(int n, T* x, int incx, T* y, int incy) {
	if (n <= 0 || incx == 0 || incy == 0) {
		return;
	}

	int ix = start_index(n, incx);
	int iy = start_index(n, incy);
	for (int i = 0; i < n; ++i) {
		std::swap(x[ix], y[iy]);
		ix += incx;
		iy += incy;
	}
}

template <typename T>
void copy_impl(int n, const T* x, int incx, T* y, int incy) {
	if (n <= 0 || incx == 0 || incy == 0) {
		return;
	}

	int ix = start_index(n, incx);
	int iy = start_index(n, incy);
	for (int i = 0; i < n; ++i) {
		y[iy] = x[ix];
		ix += incx;
		iy += incy;
	}
}

template <typename T>
void axpy_impl(int n, T alpha, const T* x, int incx, T* y, int incy) {
	if (n <= 0 || incx == 0 || incy == 0) {
		return;
	}

	int ix = start_index(n, incx);
	int iy = start_index(n, incy);
	for (int i = 0; i < n; ++i) {
		y[iy] += alpha * x[ix];
		ix += incx;
		iy += incy;
	}
}

template <typename T>
void scal_impl(int n, T alpha, T* x, int incx) {
	if (n <= 0 || incx == 0) {
		return;
	}

	int ix = start_index(n, incx);
	for (int i = 0; i < n; ++i) {
		x[ix] *= alpha;
		ix += incx;
	}
}

template <typename T>
T dot_impl(int n, const T* x, int incx, const T* y, int incy) {
	if (n <= 0 || incx == 0 || incy == 0) {
		return static_cast<T>(0);
	}

	T acc = static_cast<T>(0);
	int ix = start_index(n, incx);
	int iy = start_index(n, incy);
	for (int i = 0; i < n; ++i) {
		acc += x[ix] * y[iy];
		ix += incx;
		iy += incy;
	}
	return acc;
}

template <typename T>
std::complex<T> dotu_impl(int n, const std::complex<T>* x, int incx, const std::complex<T>* y, int incy) {
	if (n <= 0 || incx == 0 || incy == 0) {
		return std::complex<T>(0, 0);
	}

	std::complex<T> acc(0, 0);
	int ix = start_index(n, incx);
	int iy = start_index(n, incy);
	for (int i = 0; i < n; ++i) {
		acc += x[ix] * y[iy];
		ix += incx;
		iy += incy;
	}
	return acc;
}

template <typename T>
std::complex<T> dotc_impl(int n, const std::complex<T>* x, int incx, const std::complex<T>* y, int incy) {
	if (n <= 0 || incx == 0 || incy == 0) {
		return std::complex<T>(0, 0);
	}

	std::complex<T> acc(0, 0);
	int ix = start_index(n, incx);
	int iy = start_index(n, incy);
	for (int i = 0; i < n; ++i) {
		acc += std::conj(x[ix]) * y[iy];
		ix += incx;
		iy += incy;
	}
	return acc;
}

template <typename T>
T nrm2_impl(int n, const T* x, int incx) {
	if (n <= 0 || incx == 0) {
		return static_cast<T>(0);
	}

	T sum = static_cast<T>(0);
	int ix = start_index(n, incx);
	for (int i = 0; i < n; ++i) {
		const T v = x[ix];
		sum += v * v;
		ix += incx;
	}
	return std::sqrt(sum);
}

template <typename T>
T nrm2_complex_impl(int n, const std::complex<T>* x, int incx) {
	if (n <= 0 || incx == 0) {
		return static_cast<T>(0);
	}

	T sum = static_cast<T>(0);
	int ix = start_index(n, incx);
	for (int i = 0; i < n; ++i) {
		sum += std::norm(x[ix]);
		ix += incx;
	}
	return std::sqrt(sum);
}

template <typename T>
T abs1(const std::complex<T>& v) {
	return std::abs(v.real()) + std::abs(v.imag());
}

template <typename T>
T asum_impl(int n, const T* x, int incx) {
	if (n <= 0 || incx == 0) {
		return static_cast<T>(0);
	}

	T sum = static_cast<T>(0);
	int ix = start_index(n, incx);
	for (int i = 0; i < n; ++i) {
		sum += std::abs(x[ix]);
		ix += incx;
	}
	return sum;
}

template <typename T>
T asum_complex_impl(int n, const std::complex<T>* x, int incx) {
	if (n <= 0 || incx == 0) {
		return static_cast<T>(0);
	}

	T sum = static_cast<T>(0);
	int ix = start_index(n, incx);
	for (int i = 0; i < n; ++i) {
		sum += abs1(x[ix]);
		ix += incx;
	}
	return sum;
}

template <typename T>
int iamax_impl(int n, const T* x, int incx) {
	if (n <= 0 || incx <= 0) {
		return 0;
	}

	int best_logical = 0;
	int ix = 0;
	T best = std::abs(x[ix]);
	for (int i = 1; i < n; ++i) {
		ix += incx;
		const T cand = std::abs(x[ix]);
		if (cand > best) {
			best = cand;
			best_logical = i;
		}
	}

	return best_logical + 1;
}

template <typename T>
int iamax_complex_impl(int n, const std::complex<T>* x, int incx) {
	if (n <= 0 || incx <= 0) {
		return 0;
	}

	int best_logical = 0;
	int ix = 0;
	T best = abs1(x[ix]);
	for (int i = 1; i < n; ++i) {
		ix += incx;
		const T cand = abs1(x[ix]);
		if (cand > best) {
			best = cand;
			best_logical = i;
		}
	}

	return best_logical + 1;
}

template <typename T>
void rot_impl(int n, T* x, int incx, T* y, int incy, T c, T s) {
	if (n <= 0 || incx == 0 || incy == 0) {
		return;
	}

	int ix = start_index(n, incx);
	int iy = start_index(n, incy);
	for (int i = 0; i < n; ++i) {
		const T w = x[ix];
		const T z = y[iy];
		x[ix] = c * w + s * z;
		y[iy] = c * z - s * w;
		ix += incx;
		iy += incy;
	}
}

template <typename T>
void rot_complex_real_impl(int n, std::complex<T>* x, int incx, std::complex<T>* y, int incy, T c, T s) {
	if (n <= 0 || incx == 0 || incy == 0) {
		return;
	}

	int ix = start_index(n, incx);
	int iy = start_index(n, incy);
	for (int i = 0; i < n; ++i) {
		const std::complex<T> w = x[ix];
		const std::complex<T> z = y[iy];
		x[ix] = c * w + s * z;
		y[iy] = c * z - s * w;
		ix += incx;
		iy += incy;
	}
}

template <typename T>
void rotg_impl(T* a, T* b, T* c, T* s) { // NOLINT(bugprone-easily-swappable-parameters)
	const T roe = (std::abs(*a) > std::abs(*b)) ? *a : *b;
	const T scale = std::abs(*a) + std::abs(*b);
	if (scale == static_cast<T>(0)) {
		*c = static_cast<T>(1);
		*s = static_cast<T>(0);
		*a = static_cast<T>(0);
		*b = static_cast<T>(0);
		return;
	}

	T r = scale * std::sqrt((*a / scale) * (*a / scale) + (*b / scale) * (*b / scale));
	r = std::copysign(r, roe);
	*c = *a / r;
	*s = *b / r;
	T z = static_cast<T>(1);
	if (std::abs(*a) > std::abs(*b)) {
		z = *s;
	} else if (*c != static_cast<T>(0)) {
		z = static_cast<T>(1) / *c;
	}
	*a = r;
	*b = z;
}

template <typename T>
void rotg_complex_impl(std::complex<T>* a, const std::complex<T>& b, T* c, std::complex<T>* s) {
	if (std::abs(*a) == static_cast<T>(0)) {
		*c = static_cast<T>(0);
		*s = std::complex<T>(1, 0);
		*a = b;
		return;
	}

	const T scale = std::abs(*a) + std::abs(b);
	const T norm = scale * std::sqrt(std::norm(*a / scale) + std::norm(b / scale));
	const std::complex<T> alpha = *a / std::abs(*a);
	*c = std::abs(*a) / norm;
	*s = alpha * std::conj(b) / norm;
	*a = alpha * norm;
}

template <typename T>
void rotm_impl(int n, T* x, int incx, T* y, int incy, const T* p) {
	if (n <= 0 || incx == 0 || incy == 0) {
		return;
	}

	const T flag = p[0];
	if (flag == static_cast<T>(-2)) {
		return;
	}

	const T h11 = p[1];
	const T h21 = p[2];
	const T h12 = p[3];
	const T h22 = p[4];

	int ix = start_index(n, incx);
	int iy = start_index(n, incy);
	for (int i = 0; i < n; ++i) {
		const T w = x[ix];
		const T z = y[iy];
		if (flag < static_cast<T>(0)) {
			x[ix] = w * h11 + z * h12;
			y[iy] = w * h21 + z * h22;
		} else if (flag == static_cast<T>(0)) {
			x[ix] = w + z * h12;
			y[iy] = w * h21 + z;
		} else {
			x[ix] = w * h11 + z;
			y[iy] = -w + z * h22;
		}
		ix += incx;
		iy += incy;
	}
}

template <typename T>
void rotmg_impl(T* d1, T* d2, T* b1, T b2, T* p) { // NOLINT(bugprone-easily-swappable-parameters)
	if (*d1 <= static_cast<T>(0) || *d2 <= static_cast<T>(0)) {
		p[0] = static_cast<T>(-2);
		p[1] = p[2] = p[3] = p[4] = static_cast<T>(0);
		return;
	}

	T a = std::sqrt(*d1) * (*b1);
	T b = std::sqrt(*d2) * b2;
	T c = static_cast<T>(0);
	T s = static_cast<T>(0);
	rotg_impl(&a, &b, &c, &s);

	p[0] = static_cast<T>(-1);
	p[1] = c;
	p[2] = -s;
	p[3] = s;
	p[4] = c;

	const T c2 = c * c;
	const T s2 = s * s;
	const T d1_old = *d1;
	const T d2_old = *d2;
	*d1 = d1_old * c2 + d2_old * s2;
	*d2 = d1_old * s2 + d2_old * c2;
	*b1 = a / std::sqrt(std::max(*d1, static_cast<T>(1e-30)));
}

} // namespace

void sswap(int n, float* x, int incx, float* y, int incy) {
	swap_impl(n, x, incx, y, incy);
}

void dswap(int n, double* x, int incx, double* y, int incy) {
	swap_impl(n, x, incx, y, incy);
}

void cswap(int n, std::complex<float>* x, int incx, std::complex<float>* y, int incy) {
	swap_impl(n, x, incx, y, incy);
}

void zswap(int n, std::complex<double>* x, int incx, std::complex<double>* y, int incy) {
	swap_impl(n, x, incx, y, incy);
}

void scopy(int n, const float* x, int incx, float* y, int incy) {
	copy_impl(n, x, incx, y, incy);
}

void dcopy(int n, const double* x, int incx, double* y, int incy) {
	copy_impl(n, x, incx, y, incy);
}

void ccopy(int n, const std::complex<float>* x, int incx, std::complex<float>* y, int incy) {
	copy_impl(n, x, incx, y, incy);
}

void zcopy(int n, const std::complex<double>* x, int incx, std::complex<double>* y, int incy) {
	copy_impl(n, x, incx, y, incy);
}

void saxpy(int n, float alpha, const float* x, int incx, float* y, int incy) {
	axpy_impl(n, alpha, x, incx, y, incy);
}

void daxpy(int n, double alpha, const double* x, int incx, double* y, int incy) {
	axpy_impl(n, alpha, x, incx, y, incy);
}

void caxpy(int n, std::complex<float> alpha, const std::complex<float>* x, int incx, std::complex<float>* y, int incy) {
	axpy_impl(n, alpha, x, incx, y, incy);
}

void zaxpy(int n, std::complex<double> alpha, const std::complex<double>* x, int incx, std::complex<double>* y, int incy) {
	axpy_impl(n, alpha, x, incx, y, incy);
}

void sscal(int n, float alpha, float* x, int incx) {
	scal_impl(n, alpha, x, incx);
}

void dscal(int n, double alpha, double* x, int incx) {
	scal_impl(n, alpha, x, incx);
}

void cscal(int n, std::complex<float> alpha, std::complex<float>* x, int incx) {
	scal_impl(n, alpha, x, incx);
}

void zscal(int n, std::complex<double> alpha, std::complex<double>* x, int incx) {
	scal_impl(n, alpha, x, incx);
}

void csscal(int n, float alpha, std::complex<float>* x, int incx) {
	scal_impl(n, std::complex<float>(alpha, 0.0F), x, incx);
}

void zdscal(int n, double alpha, std::complex<double>* x, int incx) {
	scal_impl(n, std::complex<double>(alpha, 0.0), x, incx);
}

float sdot(int n, const float* x, int incx, const float* y, int incy) {
	return dot_impl(n, x, incx, y, incy);
}

double ddot(int n, const double* x, int incx, const double* y, int incy) {
	return dot_impl(n, x, incx, y, incy);
}

std::complex<float> cdotu(int n, const std::complex<float>* x, int incx, const std::complex<float>* y, int incy) {
	return dotu_impl(n, x, incx, y, incy);
}

std::complex<float> cdotc(int n, const std::complex<float>* x, int incx, const std::complex<float>* y, int incy) {
	return dotc_impl(n, x, incx, y, incy);
}

std::complex<double> zdotu(int n, const std::complex<double>* x, int incx, const std::complex<double>* y, int incy) {
	return dotu_impl(n, x, incx, y, incy);
}

std::complex<double> zdotc(int n, const std::complex<double>* x, int incx, const std::complex<double>* y, int incy) {
	return dotc_impl(n, x, incx, y, incy);
}

float snrm2(int n, const float* x, int incx) {
	return nrm2_impl(n, x, incx);
}

double dnrm2(int n, const double* x, int incx) {
	return nrm2_impl(n, x, incx);
}

float scnrm2(int n, const std::complex<float>* x, int incx) {
	return nrm2_complex_impl(n, x, incx);
}

double dznrm2(int n, const std::complex<double>* x, int incx) {
	return nrm2_complex_impl(n, x, incx);
}

float sasum(int n, const float* x, int incx) {
	return asum_impl(n, x, incx);
}

double dasum(int n, const double* x, int incx) {
	return asum_impl(n, x, incx);
}

float scasum(int n, const std::complex<float>* x, int incx) {
	return asum_complex_impl(n, x, incx);
}

double dzasum(int n, const std::complex<double>* x, int incx) {
	return asum_complex_impl(n, x, incx);
}

void srot(int n, float* x, int incx, float* y, int incy, float c, float s) {
	rot_impl(n, x, incx, y, incy, c, s);
}

void drot(int n, double* x, int incx, double* y, int incy, double c, double s) {
	rot_impl(n, x, incx, y, incy, c, s);
}

void csrot(int n, std::complex<float>* x, int incx, std::complex<float>* y, int incy, float c, float s) {
	rot_complex_real_impl(n, x, incx, y, incy, c, s);
}

void zdrot(int n, std::complex<double>* x, int incx, std::complex<double>* y, int incy, double c, double s) {
	rot_complex_real_impl(n, x, incx, y, incy, c, s);
}

void srotg(float* a, float* b, float* c, float* s) {
	rotg_impl(a, b, c, s);
}

void drotg(double* a, double* b, double* c, double* s) {
	rotg_impl(a, b, c, s);
}

void crotg(std::complex<float>* a, std::complex<float> b, float* c, std::complex<float>* s) {
	rotg_complex_impl(a, b, c, s);
}

void zrotg(std::complex<double>* a, std::complex<double> b, double* c, std::complex<double>* s) {
	rotg_complex_impl(a, b, c, s);
}

void srotm(int n, float* x, int incx, float* y, int incy, const float* param) {
	rotm_impl(n, x, incx, y, incy, param);
}

void drotm(int n, double* x, int incx, double* y, int incy, const double* param) {
	rotm_impl(n, x, incx, y, incy, param);
}

void srotmg(float* d1, float* d2, float* b1, float b2, float* param) {
	rotmg_impl(d1, d2, b1, b2, param);
}

void drotmg(double* d1, double* d2, double* b1, double b2, double* param) {
	rotmg_impl(d1, d2, b1, b2, param);
}

int isamax(int n, const float* x, int incx) {
	return iamax_impl(n, x, incx);
}

int idamax(int n, const double* x, int incx) {
	return iamax_impl(n, x, incx);
}

int icamax(int n, const std::complex<float>* x, int incx) {
	return iamax_complex_impl(n, x, incx);
}

int izamax(int n, const std::complex<double>* x, int incx) {
	return iamax_complex_impl(n, x, incx);
}

} // namespace theblas
