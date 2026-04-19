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

	auto acc = static_cast<T>(0);
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

	auto sum = static_cast<T>(0);
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

	auto sum = static_cast<T>(0);
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

	auto sum = static_cast<T>(0);
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

	auto sum = static_cast<T>(0);
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
	auto z = static_cast<T>(1);
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
	auto c = static_cast<T>(0);
	auto s = static_cast<T>(0);
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

/* ====================================================================
 * Level 2 BLAS — Matrix-Vector Operations
 * ==================================================================== */

namespace {

inline char to_upper(char c) {
	return static_cast<char>(c >= 'a' && c <= 'z' ? c - 32 : c);
}

inline bool valid_trans(char t) {
	char u = to_upper(t);
	return u == 'N' || u == 'T' || u == 'C';
}

inline bool valid_uplo(char u) {
	char c = to_upper(u);
	return c == 'U' || c == 'L';
}

inline bool valid_diag(char d) {
	char c = to_upper(d);
	return c == 'U' || c == 'N';
}

template <typename T>
T conj_val(T v) {
	return v;
}

template <typename T>
std::complex<T> conj_val(std::complex<T> v) {
	return std::conj(v);
}

/* ------------------------------------------------------------------ */
/* gemv_impl — y ← α·op(A)·x + β·y                                  */
/* ------------------------------------------------------------------ */
template <typename T>
void gemv_impl(char trans, int m, int n, T alpha, const T* A, int lda, const T* x, int incx,
               T beta, T* y, int incy) {
	char tr = to_upper(trans);
	if (!valid_trans(trans) || m <= 0 || n <= 0 || incx == 0 || incy == 0) {
		return;
	}
	int leny = (tr == 'N') ? m : n;
	int lenx = (tr == 'N') ? n : m;

	// Scale y by beta
	{
		int iy = start_index(leny, incy);
		if (beta == T(0)) {
			for (int i = 0; i < leny; ++i) {
				y[iy] = T(0);
				iy += incy;
			}
		} else if (beta != T(1)) {
			for (int i = 0; i < leny; ++i) {
				y[iy] *= beta;
				iy += incy;
			}
		}
	}

	if (alpha == T(0)) {
		return;
	}

	if (tr == 'N') {
		// y ← α·A·x + y
		int jx = start_index(lenx, incx);
		for (int j = 0; j < n; ++j) {
			T temp = alpha * x[jx];
			int iy = start_index(leny, incy);
			for (int i = 0; i < m; ++i) {
				y[iy] += temp * A[i + j * lda];
				iy += incy;
			}
			jx += incx;
		}
	} else if (tr == 'T') {
		// y ← α·Aᵀ·x + y
		int jy = start_index(leny, incy);
		for (int j = 0; j < n; ++j) {
			T temp = T(0);
			int ix = start_index(lenx, incx);
			for (int i = 0; i < m; ++i) {
				temp += A[i + j * lda] * x[ix];
				ix += incx;
			}
			y[jy] += alpha * temp;
			jy += incy;
		}
	} else {
		// tr == 'C': y ← α·Aᴴ·x + y (conjugate-transpose)
		int jy = start_index(leny, incy);
		for (int j = 0; j < n; ++j) {
			T temp = T(0);
			int ix = start_index(lenx, incx);
			for (int i = 0; i < m; ++i) {
				temp += conj_val(A[i + j * lda]) * x[ix];
				ix += incx;
			}
			y[jy] += alpha * temp;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* symv_impl — y ← α·A·x + β·y   (A symmetric)                     */
/* ------------------------------------------------------------------ */
template <typename T>
void symv_impl(char uplo, int n, T alpha, const T* A, int lda, const T* x, int incx, T beta,
               T* y, int incy) {
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0) {
		return;
	}

	// Scale y by beta
	{
		int iy = start_index(n, incy);
		if (beta == T(0)) {
			for (int i = 0; i < n; ++i) {
				y[iy] = T(0);
				iy += incy;
			}
		} else if (beta != T(1)) {
			for (int i = 0; i < n; ++i) {
				y[iy] *= beta;
				iy += incy;
			}
		}
	}

	if (alpha == T(0)) {
		return;
	}

	if (ul == 'U') {
		int jx = start_index(n, incx);
		int jy = start_index(n, incy);
		for (int j = 0; j < n; ++j) {
			T temp1 = alpha * x[jx];
			T temp2 = T(0);
			int ix = start_index(n, incx);
			int iy = start_index(n, incy);
			for (int i = 0; i < j; ++i) {
				y[iy] += temp1 * A[i + j * lda];
				temp2 += A[i + j * lda] * x[ix];
				ix += incx;
				iy += incy;
			}
			y[jy] += temp1 * A[j + j * lda] + alpha * temp2;
			jx += incx;
			jy += incy;
		}
	} else {
		int jx = start_index(n, incx);
		int jy = start_index(n, incy);
		for (int j = 0; j < n; ++j) {
			T temp1 = alpha * x[jx];
			T temp2 = T(0);
			y[jy] += temp1 * A[j + j * lda];
			int ix = jx;
			int iy = jy;
			for (int i = j + 1; i < n; ++i) {
				ix += incx;
				iy += incy;
				y[iy] += temp1 * A[i + j * lda];
				temp2 += A[i + j * lda] * x[ix];
			}
			y[jy] += alpha * temp2;
			jx += incx;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* hemv_impl — y ← α·A·x + β·y   (A Hermitian)                     */
/* ------------------------------------------------------------------ */
template <typename T>
void hemv_impl(char uplo, int n, std::complex<T> alpha, const std::complex<T>* A, int lda,
               const std::complex<T>* x, int incx, std::complex<T> beta, std::complex<T>* y,
               int incy) {
	using C = std::complex<T>;
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0) {
		return;
	}

	// Scale y by beta
	{
		int iy = start_index(n, incy);
		if (beta == C(0)) {
			for (int i = 0; i < n; ++i) {
				y[iy] = C(0);
				iy += incy;
			}
		} else if (beta != C(1)) {
			for (int i = 0; i < n; ++i) {
				y[iy] *= beta;
				iy += incy;
			}
		}
	}

	if (alpha == C(0)) {
		return;
	}

	if (ul == 'U') {
		int jx = start_index(n, incx);
		int jy = start_index(n, incy);
		for (int j = 0; j < n; ++j) {
			C temp1 = alpha * x[jx];
			C temp2 = C(0);
			int ix = start_index(n, incx);
			int iy = start_index(n, incy);
			for (int i = 0; i < j; ++i) {
				y[iy] += temp1 * A[i + j * lda];
				temp2 += std::conj(A[i + j * lda]) * x[ix];
				ix += incx;
				iy += incy;
			}
			// Diagonal of a Hermitian matrix is real
			y[jy] += temp1 * T(A[j + j * lda].real()) + alpha * temp2;
			jx += incx;
			jy += incy;
		}
	} else {
		int jx = start_index(n, incx);
		int jy = start_index(n, incy);
		for (int j = 0; j < n; ++j) {
			C temp1 = alpha * x[jx];
			C temp2 = C(0);
			y[jy] += temp1 * T(A[j + j * lda].real());
			int ix = jx;
			int iy = jy;
			for (int i = j + 1; i < n; ++i) {
				ix += incx;
				iy += incy;
				y[iy] += temp1 * A[i + j * lda];
				temp2 += std::conj(A[i + j * lda]) * x[ix];
			}
			y[jy] += alpha * temp2;
			jx += incx;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* trmv_impl — x ← op(A)·x   (A triangular)                        */
/* ------------------------------------------------------------------ */
template <typename T>
void trmv_impl(char uplo, char trans, char diag, int n, const T* A, int lda, T* x, int incx) {
	char ul = to_upper(uplo);
	char tr = to_upper(trans);
	char dg = to_upper(diag);
	if (!valid_uplo(uplo) || !valid_trans(trans) || !valid_diag(diag) || n <= 0 || incx == 0) {
		return;
	}
	bool unit = (dg == 'U');

	if (tr == 'N') {
		if (ul == 'U') {
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				if (x[jx] != T(0)) {
					T temp = x[jx];
					int ix = start_index(n, incx);
					for (int i = 0; i < j; ++i) {
						x[ix] += temp * A[i + j * lda];
						ix += incx;
					}
					if (!unit) {
						x[jx] *= A[j + j * lda];
					}
				}
				jx += incx;
			}
		} else {
			int jx = start_index(n, incx) + (n - 1) * incx;
			for (int j = n - 1; j >= 0; --j) {
				if (x[jx] != T(0)) {
					T temp = x[jx];
					int ix = start_index(n, incx) + (n - 1) * incx;
					for (int i = n - 1; i > j; --i) {
						x[ix] += temp * A[i + j * lda];
						ix -= incx;
					}
					if (!unit) {
						x[jx] *= A[j + j * lda];
					}
				}
				jx -= incx;
			}
		}
	} else {
		// Transpose or conjugate-transpose
		bool conj = (tr == 'C');
		if (ul == 'U') {
			int jx = start_index(n, incx) + (n - 1) * incx;
			for (int j = n - 1; j >= 0; --j) {
				T temp = x[jx];
				if (!unit) {
					temp *= conj ? conj_val(A[j + j * lda]) : A[j + j * lda];
				}
				int ix = jx;
				for (int i = j - 1; i >= 0; --i) {
					ix -= incx;
					temp += (conj ? conj_val(A[i + j * lda]) : A[i + j * lda]) * x[ix];
				}
				x[jx] = temp;
				jx -= incx;
			}
		} else {
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				T temp = x[jx];
				if (!unit) {
					temp *= conj ? conj_val(A[j + j * lda]) : A[j + j * lda];
				}
				int ix = jx;
				for (int i = j + 1; i < n; ++i) {
					ix += incx;
					temp += (conj ? conj_val(A[i + j * lda]) : A[i + j * lda]) * x[ix];
				}
				x[jx] = temp;
				jx += incx;
			}
		}
	}
}

/* ------------------------------------------------------------------ */
/* trsv_impl — solve op(A)·x = b   (A triangular)                   */
/* ------------------------------------------------------------------ */
template <typename T>
void trsv_impl(char uplo, char trans, char diag, int n, const T* A, int lda, T* x, int incx) {
	char ul = to_upper(uplo);
	char tr = to_upper(trans);
	char dg = to_upper(diag);
	if (!valid_uplo(uplo) || !valid_trans(trans) || !valid_diag(diag) || n <= 0 || incx == 0) {
		return;
	}
	bool unit = (dg == 'U');

	if (tr == 'N') {
		if (ul == 'U') {
			// Back-substitution
			int jx = start_index(n, incx) + (n - 1) * incx;
			for (int j = n - 1; j >= 0; --j) {
				if (!unit) {
					x[jx] /= A[j + j * lda];
				}
				T temp = x[jx];
				int ix = jx;
				for (int i = j - 1; i >= 0; --i) {
					ix -= incx;
					x[ix] -= temp * A[i + j * lda];
				}
				jx -= incx;
			}
		} else {
			// Forward-substitution
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				if (!unit) {
					x[jx] /= A[j + j * lda];
				}
				T temp = x[jx];
				int ix = jx;
				for (int i = j + 1; i < n; ++i) {
					ix += incx;
					x[ix] -= temp * A[i + j * lda];
				}
				jx += incx;
			}
		}
	} else {
		bool conj = (tr == 'C');
		if (ul == 'U') {
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				T temp = x[jx];
				int ix = start_index(n, incx);
				for (int i = 0; i < j; ++i) {
					temp -= (conj ? conj_val(A[i + j * lda]) : A[i + j * lda]) * x[ix];
					ix += incx;
				}
				if (!unit) {
					temp /= conj ? conj_val(A[j + j * lda]) : A[j + j * lda];
				}
				x[jx] = temp;
				jx += incx;
			}
		} else {
			int jx = start_index(n, incx) + (n - 1) * incx;
			for (int j = n - 1; j >= 0; --j) {
				T temp = x[jx];
				int ix = start_index(n, incx) + (n - 1) * incx;
				for (int i = n - 1; i > j; --i) {
					temp -= (conj ? conj_val(A[i + j * lda]) : A[i + j * lda]) * x[ix];
					ix -= incx;
				}
				if (!unit) {
					temp /= conj ? conj_val(A[j + j * lda]) : A[j + j * lda];
				}
				x[jx] = temp;
				jx -= incx;
			}
		}
	}
}

/* ------------------------------------------------------------------ */
/* ger_impl — A ← α·x·yᵀ + A   (real rank-1 update)                */
/* ------------------------------------------------------------------ */
template <typename T>
void ger_impl(int m, int n, T alpha, const T* x, int incx, const T* y, int incy, T* A, int lda) {
	if (m <= 0 || n <= 0 || incx == 0 || incy == 0) {
		return;
	}
	if (alpha == T(0)) {
		return;
	}

	int jy = start_index(n, incy);
	for (int j = 0; j < n; ++j) {
		T temp = alpha * y[jy];
		int ix = start_index(m, incx);
		for (int i = 0; i < m; ++i) {
			A[i + j * lda] += x[ix] * temp;
			ix += incx;
		}
		jy += incy;
	}
}

/* ------------------------------------------------------------------ */
/* geru_impl — A ← α·x·yᵀ + A   (complex unconjugated rank-1)      */
/* ------------------------------------------------------------------ */
template <typename T>
void geru_impl(int m, int n, std::complex<T> alpha, const std::complex<T>* x, int incx,
               const std::complex<T>* y, int incy, std::complex<T>* A, int lda) {
	using C = std::complex<T>;
	if (m <= 0 || n <= 0 || incx == 0 || incy == 0) {
		return;
	}
	if (alpha == C(0)) {
		return;
	}

	int jy = start_index(n, incy);
	for (int j = 0; j < n; ++j) {
		C temp = alpha * y[jy];
		int ix = start_index(m, incx);
		for (int i = 0; i < m; ++i) {
			A[i + j * lda] += x[ix] * temp;
			ix += incx;
		}
		jy += incy;
	}
}

/* ------------------------------------------------------------------ */
/* gerc_impl — A ← α·x·conj(y)ᵀ + A   (complex conjugated rank-1)  */
/* ------------------------------------------------------------------ */
template <typename T>
void gerc_impl(int m, int n, std::complex<T> alpha, const std::complex<T>* x, int incx,
               const std::complex<T>* y, int incy, std::complex<T>* A, int lda) {
	using C = std::complex<T>;
	if (m <= 0 || n <= 0 || incx == 0 || incy == 0) {
		return;
	}
	if (alpha == C(0)) {
		return;
	}

	int jy = start_index(n, incy);
	for (int j = 0; j < n; ++j) {
		C temp = alpha * std::conj(y[jy]);
		int ix = start_index(m, incx);
		for (int i = 0; i < m; ++i) {
			A[i + j * lda] += x[ix] * temp;
			ix += incx;
		}
		jy += incy;
	}
}

/* ------------------------------------------------------------------ */
/* syr_impl — A ← α·x·xᵀ + A   (A symmetric)                       */
/* ------------------------------------------------------------------ */
template <typename T>
void syr_impl(char uplo, int n, T alpha, const T* x, int incx, T* A, int lda) {
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || incx == 0) {
		return;
	}
	if (alpha == T(0)) {
		return;
	}

	int jx = start_index(n, incx);
	if (ul == 'U') {
		for (int j = 0; j < n; ++j) {
			T temp = alpha * x[jx];
			int ix = start_index(n, incx);
			for (int i = 0; i <= j; ++i) {
				A[i + j * lda] += x[ix] * temp;
				ix += incx;
			}
			jx += incx;
		}
	} else {
		for (int j = 0; j < n; ++j) {
			T temp = alpha * x[jx];
			int ix = jx;
			for (int i = j; i < n; ++i) {
				A[i + j * lda] += x[ix] * temp;
				ix += incx;
			}
			jx += incx;
		}
	}
}

/* ------------------------------------------------------------------ */
/* her_impl — A ← α·x·xᴴ + A   (A Hermitian, α real)               */
/* ------------------------------------------------------------------ */
template <typename T>
void her_impl(char uplo, int n, T alpha, const std::complex<T>* x, int incx,
              std::complex<T>* A, int lda) {
	using C = std::complex<T>;
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || incx == 0) {
		return;
	}
	if (alpha == T(0)) {
		return;
	}

	int jx = start_index(n, incx);
	if (ul == 'U') {
		for (int j = 0; j < n; ++j) {
			C temp = C(alpha) * std::conj(x[jx]);
			int ix = start_index(n, incx);
			for (int i = 0; i < j; ++i) {
				A[i + j * lda] += x[ix] * temp;
				ix += incx;
			}
			// Diagonal is real
			A[j + j * lda] = C(A[j + j * lda].real() + (x[jx] * temp).real());
			jx += incx;
		}
	} else {
		for (int j = 0; j < n; ++j) {
			C temp = C(alpha) * std::conj(x[jx]);
			// Diagonal is real
			A[j + j * lda] = C(A[j + j * lda].real() + (x[jx] * temp).real());
			int ix = jx;
			for (int i = j + 1; i < n; ++i) {
				ix += incx;
				A[i + j * lda] += x[ix] * temp;
			}
			jx += incx;
		}
	}
}

/* ------------------------------------------------------------------ */
/* syr2_impl — A ← α·x·yᵀ + α·y·xᵀ + A   (A symmetric)            */
/* ------------------------------------------------------------------ */
template <typename T>
void syr2_impl(char uplo, int n, T alpha, const T* x, int incx, const T* y, int incy, T* A,
               int lda) {
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0) {
		return;
	}
	if (alpha == T(0)) {
		return;
	}

	int jx = start_index(n, incx);
	int jy = start_index(n, incy);
	if (ul == 'U') {
		for (int j = 0; j < n; ++j) {
			T temp1 = alpha * y[jy];
			T temp2 = alpha * x[jx];
			int ix = start_index(n, incx);
			int iy = start_index(n, incy);
			for (int i = 0; i <= j; ++i) {
				A[i + j * lda] += x[ix] * temp1 + y[iy] * temp2;
				ix += incx;
				iy += incy;
			}
			jx += incx;
			jy += incy;
		}
	} else {
		for (int j = 0; j < n; ++j) {
			T temp1 = alpha * y[jy];
			T temp2 = alpha * x[jx];
			int ix = jx;
			int iy = jy;
			for (int i = j; i < n; ++i) {
				A[i + j * lda] += x[ix] * temp1 + y[iy] * temp2;
				ix += incx;
				iy += incy;
			}
			jx += incx;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* her2_impl — A ← α·x·yᴴ + conj(α)·y·xᴴ + A   (A Hermitian)      */
/* ------------------------------------------------------------------ */
template <typename T>
void her2_impl(char uplo, int n, std::complex<T> alpha, const std::complex<T>* x, int incx,
               const std::complex<T>* y, int incy, std::complex<T>* A, int lda) {
	using C = std::complex<T>;
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0) {
		return;
	}
	if (alpha == C(0)) {
		return;
	}

	int jx = start_index(n, incx);
	int jy = start_index(n, incy);
	if (ul == 'U') {
		for (int j = 0; j < n; ++j) {
			C temp1 = alpha * std::conj(y[jy]);
			C temp2 = std::conj(alpha * x[jx]);
			int ix = start_index(n, incx);
			int iy = start_index(n, incy);
			for (int i = 0; i < j; ++i) {
				A[i + j * lda] += x[ix] * temp1 + y[iy] * temp2;
				ix += incx;
				iy += incy;
			}
			A[j + j * lda] =
			    C((A[j + j * lda] + x[jx] * temp1 + y[jy] * temp2).real());
			jx += incx;
			jy += incy;
		}
	} else {
		for (int j = 0; j < n; ++j) {
			C temp1 = alpha * std::conj(y[jy]);
			C temp2 = std::conj(alpha * x[jx]);
			A[j + j * lda] =
			    C((A[j + j * lda] + x[jx] * temp1 + y[jy] * temp2).real());
			int ix = jx;
			int iy = jy;
			for (int i = j + 1; i < n; ++i) {
				ix += incx;
				iy += incy;
				A[i + j * lda] += x[ix] * temp1 + y[iy] * temp2;
			}
			jx += incx;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* gbmv_impl — y ← α·op(A)·x + β·y   (A banded)                    */
/* ------------------------------------------------------------------ */
template <typename T>
void gbmv_impl(char trans, int m, int n, int kl, int ku, T alpha, const T* A, int lda,
               const T* x, int incx, T beta, T* y, int incy) {
	char tr = to_upper(trans);
	if (!valid_trans(trans) || m <= 0 || n <= 0 || kl < 0 || ku < 0 || incx == 0 || incy == 0) {
		return;
	}
	int leny = (tr == 'N') ? m : n;
	int lenx = (tr == 'N') ? n : m;

	// Scale y by beta
	{
		int iy = start_index(leny, incy);
		if (beta == T(0)) {
			for (int i = 0; i < leny; ++i) {
				y[iy] = T(0);
				iy += incy;
			}
		} else if (beta != T(1)) {
			for (int i = 0; i < leny; ++i) {
				y[iy] *= beta;
				iy += incy;
			}
		}
	}

	if (alpha == T(0)) {
		return;
	}

	if (tr == 'N') {
		int jx = start_index(lenx, incx);
		for (int j = 0; j < n; ++j) {
			T temp = alpha * x[jx];
			int i_start = std::max(0, j - ku);
			int i_end = std::min(m - 1, j + kl);
			int iy = start_index(leny, incy) + i_start * incy;
			for (int i = i_start; i <= i_end; ++i) {
				y[iy] += temp * A[ku + i - j + j * lda];
				iy += incy;
			}
			jx += incx;
		}
	} else if (tr == 'T') {
		int jy = start_index(leny, incy);
		for (int j = 0; j < n; ++j) {
			T temp = T(0);
			int i_start = std::max(0, j - ku);
			int i_end = std::min(m - 1, j + kl);
			int ix = start_index(lenx, incx) + i_start * incx;
			for (int i = i_start; i <= i_end; ++i) {
				temp += A[ku + i - j + j * lda] * x[ix];
				ix += incx;
			}
			y[jy] += alpha * temp;
			jy += incy;
		}
	} else {
		// 'C' conjugate-transpose
		int jy = start_index(leny, incy);
		for (int j = 0; j < n; ++j) {
			T temp = T(0);
			int i_start = std::max(0, j - ku);
			int i_end = std::min(m - 1, j + kl);
			int ix = start_index(lenx, incx) + i_start * incx;
			for (int i = i_start; i <= i_end; ++i) {
				temp += conj_val(A[ku + i - j + j * lda]) * x[ix];
				ix += incx;
			}
			y[jy] += alpha * temp;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* sbmv_impl — y ← α·A·x + β·y   (A symmetric banded)              */
/* ------------------------------------------------------------------ */
template <typename T>
void sbmv_impl(char uplo, int n, int k, T alpha, const T* A, int lda, const T* x, int incx,
               T beta, T* y, int incy) {
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || k < 0 || incx == 0 || incy == 0) {
		return;
	}

	// Scale y by beta
	{
		int iy = start_index(n, incy);
		if (beta == T(0)) {
			for (int i = 0; i < n; ++i) {
				y[iy] = T(0);
				iy += incy;
			}
		} else if (beta != T(1)) {
			for (int i = 0; i < n; ++i) {
				y[iy] *= beta;
				iy += incy;
			}
		}
	}

	if (alpha == T(0)) {
		return;
	}

	if (ul == 'U') {
		int jx = start_index(n, incx);
		int jy = start_index(n, incy);
		for (int j = 0; j < n; ++j) {
			T temp1 = alpha * x[jx];
			T temp2 = T(0);
			int l = k - j;
			int i_start = std::max(0, j - k);
			int ix = start_index(n, incx) + i_start * incx;
			int iy = start_index(n, incy) + i_start * incy;
			for (int i = i_start; i < j; ++i) {
				y[iy] += temp1 * A[l + i + j * lda];
				temp2 += A[l + i + j * lda] * x[ix];
				ix += incx;
				iy += incy;
			}
			y[jy] += temp1 * A[k + j * lda] + alpha * temp2;
			jx += incx;
			jy += incy;
		}
	} else {
		int jx = start_index(n, incx);
		int jy = start_index(n, incy);
		for (int j = 0; j < n; ++j) {
			T temp1 = alpha * x[jx];
			T temp2 = T(0);
			y[jy] += temp1 * A[0 + j * lda];
			int ix = jx;
			int iy = jy;
			int i_end = std::min(n - 1, j + k);
			for (int i = j + 1; i <= i_end; ++i) {
				ix += incx;
				iy += incy;
				y[iy] += temp1 * A[i - j + j * lda];
				temp2 += A[i - j + j * lda] * x[ix];
			}
			y[jy] += alpha * temp2;
			jx += incx;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* hbmv_impl — y ← α·A·x + β·y   (A Hermitian banded)              */
/* ------------------------------------------------------------------ */
template <typename T>
void hbmv_impl(char uplo, int n, int k, std::complex<T> alpha, const std::complex<T>* A,
               int lda, const std::complex<T>* x, int incx, std::complex<T> beta,
               std::complex<T>* y, int incy) {
	using C = std::complex<T>;
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || k < 0 || incx == 0 || incy == 0) {
		return;
	}

	// Scale y by beta
	{
		int iy = start_index(n, incy);
		if (beta == C(0)) {
			for (int i = 0; i < n; ++i) {
				y[iy] = C(0);
				iy += incy;
			}
		} else if (beta != C(1)) {
			for (int i = 0; i < n; ++i) {
				y[iy] *= beta;
				iy += incy;
			}
		}
	}

	if (alpha == C(0)) {
		return;
	}

	if (ul == 'U') {
		int jx = start_index(n, incx);
		int jy = start_index(n, incy);
		for (int j = 0; j < n; ++j) {
			C temp1 = alpha * x[jx];
			C temp2 = C(0);
			int l = k - j;
			int i_start = std::max(0, j - k);
			int ix = start_index(n, incx) + i_start * incx;
			int iy = start_index(n, incy) + i_start * incy;
			for (int i = i_start; i < j; ++i) {
				y[iy] += temp1 * A[l + i + j * lda];
				temp2 += std::conj(A[l + i + j * lda]) * x[ix];
				ix += incx;
				iy += incy;
			}
			y[jy] += temp1 * T(A[k + j * lda].real()) + alpha * temp2;
			jx += incx;
			jy += incy;
		}
	} else {
		int jx = start_index(n, incx);
		int jy = start_index(n, incy);
		for (int j = 0; j < n; ++j) {
			C temp1 = alpha * x[jx];
			C temp2 = C(0);
			y[jy] += temp1 * T(A[0 + j * lda].real());
			int ix = jx;
			int iy = jy;
			int i_end = std::min(n - 1, j + k);
			for (int i = j + 1; i <= i_end; ++i) {
				ix += incx;
				iy += incy;
				y[iy] += temp1 * A[i - j + j * lda];
				temp2 += std::conj(A[i - j + j * lda]) * x[ix];
			}
			y[jy] += alpha * temp2;
			jx += incx;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* tbmv_impl — x ← op(A)·x   (A triangular banded)                  */
/* ------------------------------------------------------------------ */
template <typename T>
void tbmv_impl(char uplo, char trans, char diag, int n, int k, const T* A, int lda, T* x,
               int incx) {
	char ul = to_upper(uplo);
	char tr = to_upper(trans);
	char dg = to_upper(diag);
	if (!valid_uplo(uplo) || !valid_trans(trans) || !valid_diag(diag) || n <= 0 || k < 0 ||
	    incx == 0) {
		return;
	}
	bool unit = (dg == 'U');
	bool conj = (tr == 'C');

	if (tr == 'N') {
		if (ul == 'U') {
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				if (x[jx] != T(0)) {
					T temp = x[jx];
					int l = k - j;
					int i_start = std::max(0, j - k);
					int ix = start_index(n, incx) + i_start * incx;
					for (int i = i_start; i < j; ++i) {
						x[ix] += temp * A[l + i + j * lda];
						ix += incx;
					}
					if (!unit) {
						x[jx] *= A[k + j * lda];
					}
				}
				jx += incx;
			}
		} else {
			int jx = start_index(n, incx) + (n - 1) * incx;
			for (int j = n - 1; j >= 0; --j) {
				if (x[jx] != T(0)) {
					T temp = x[jx];
					int i_end = std::min(n - 1, j + k);
					int ix = start_index(n, incx) + i_end * incx;
					for (int i = i_end; i > j; --i) {
						x[ix] += temp * A[i - j + j * lda];
						ix -= incx;
					}
					if (!unit) {
						x[jx] *= A[0 + j * lda];
					}
				}
				jx -= incx;
			}
		}
	} else {
		// Transpose or conjugate-transpose
		if (ul == 'U') {
			int jx = start_index(n, incx) + (n - 1) * incx;
			for (int j = n - 1; j >= 0; --j) {
				T temp = x[jx];
				int l = k - j;
				if (!unit) {
					T aval = conj ? conj_val(A[k + j * lda]) : A[k + j * lda];
					temp *= aval;
				}
				int i_start = std::max(0, j - k);
				int ix = jx;
				for (int i = j - 1; i >= i_start; --i) {
					ix -= incx;
					T aval = conj ? conj_val(A[l + i + j * lda]) : A[l + i + j * lda];
					temp += aval * x[ix];
				}
				x[jx] = temp;
				jx -= incx;
			}
		} else {
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				T temp = x[jx];
				if (!unit) {
					T aval = conj ? conj_val(A[0 + j * lda]) : A[0 + j * lda];
					temp *= aval;
				}
				int i_end = std::min(n - 1, j + k);
				int ix = jx;
				for (int i = j + 1; i <= i_end; ++i) {
					ix += incx;
					T aval = conj ? conj_val(A[i - j + j * lda]) : A[i - j + j * lda];
					temp += aval * x[ix];
				}
				x[jx] = temp;
				jx += incx;
			}
		}
	}
}

/* ------------------------------------------------------------------ */
/* tbsv_impl — solve op(A)·x = b   (A triangular banded)             */
/* ------------------------------------------------------------------ */
template <typename T>
void tbsv_impl(char uplo, char trans, char diag, int n, int k, const T* A, int lda, T* x,
               int incx) {
	char ul = to_upper(uplo);
	char tr = to_upper(trans);
	char dg = to_upper(diag);
	if (!valid_uplo(uplo) || !valid_trans(trans) || !valid_diag(diag) || n <= 0 || k < 0 ||
	    incx == 0) {
		return;
	}
	bool unit = (dg == 'U');
	bool conj = (tr == 'C');

	if (tr == 'N') {
		if (ul == 'U') {
			int jx = start_index(n, incx) + (n - 1) * incx;
			for (int j = n - 1; j >= 0; --j) {
				if (!unit) {
					x[jx] /= A[k + j * lda];
				}
				T temp = x[jx];
				int l = k - j;
				int i_start = std::max(0, j - k);
				int ix = jx;
				for (int i = j - 1; i >= i_start; --i) {
					ix -= incx;
					x[ix] -= temp * A[l + i + j * lda];
				}
				jx -= incx;
			}
		} else {
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				if (!unit) {
					x[jx] /= A[0 + j * lda];
				}
				T temp = x[jx];
				int i_end = std::min(n - 1, j + k);
				int ix = jx;
				for (int i = j + 1; i <= i_end; ++i) {
					ix += incx;
					x[ix] -= temp * A[i - j + j * lda];
				}
				jx += incx;
			}
		}
	} else {
		if (ul == 'U') {
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				T temp = x[jx];
				int l = k - j;
				int i_start = std::max(0, j - k);
				int ix = start_index(n, incx) + i_start * incx;
				for (int i = i_start; i < j; ++i) {
					T aval = conj ? conj_val(A[l + i + j * lda]) : A[l + i + j * lda];
					temp -= aval * x[ix];
					ix += incx;
				}
				if (!unit) {
					T aval = conj ? conj_val(A[k + j * lda]) : A[k + j * lda];
					temp /= aval;
				}
				x[jx] = temp;
				jx += incx;
			}
		} else {
			int jx = start_index(n, incx) + (n - 1) * incx;
			for (int j = n - 1; j >= 0; --j) {
				T temp = x[jx];
				int i_end = std::min(n - 1, j + k);
				int ix = start_index(n, incx) + i_end * incx;
				for (int i = i_end; i > j; --i) {
					T aval = conj ? conj_val(A[i - j + j * lda]) : A[i - j + j * lda];
					temp -= aval * x[ix];
					ix -= incx;
				}
				if (!unit) {
					T aval = conj ? conj_val(A[0 + j * lda]) : A[0 + j * lda];
					temp /= aval;
				}
				x[jx] = temp;
				jx -= incx;
			}
		}
	}
}

/* ------------------------------------------------------------------ */
/* spmv_impl — y ← α·A·x + β·y   (A symmetric, packed)             */
/* ------------------------------------------------------------------ */
template <typename T>
void spmv_impl(char uplo, int n, T alpha, const T* Ap, const T* x, int incx, T beta, T* y,
               int incy) {
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0) {
		return;
	}

	// Scale y by beta
	{
		int iy = start_index(n, incy);
		if (beta == T(0)) {
			for (int i = 0; i < n; ++i) {
				y[iy] = T(0);
				iy += incy;
			}
		} else if (beta != T(1)) {
			for (int i = 0; i < n; ++i) {
				y[iy] *= beta;
				iy += incy;
			}
		}
	}

	if (alpha == T(0)) {
		return;
	}

	int kk = 0;
	if (ul == 'U') {
		int jx = start_index(n, incx);
		int jy = start_index(n, incy);
		for (int j = 0; j < n; ++j) {
			T temp1 = alpha * x[jx];
			T temp2 = T(0);
			int ix = start_index(n, incx);
			int iy = start_index(n, incy);
			for (int i = 0; i < j; ++i) {
				y[iy] += temp1 * Ap[kk + i];
				temp2 += Ap[kk + i] * x[ix];
				ix += incx;
				iy += incy;
			}
			y[jy] += temp1 * Ap[kk + j] + alpha * temp2;
			kk += j + 1;
			jx += incx;
			jy += incy;
		}
	} else {
		int jx = start_index(n, incx);
		int jy = start_index(n, incy);
		for (int j = 0; j < n; ++j) {
			T temp1 = alpha * x[jx];
			T temp2 = T(0);
			y[jy] += temp1 * Ap[kk];
			int ix = jx;
			int iy = jy;
			for (int i = j + 1; i < n; ++i) {
				ix += incx;
				iy += incy;
				y[iy] += temp1 * Ap[kk + i - j];
				temp2 += Ap[kk + i - j] * x[ix];
			}
			y[jy] += alpha * temp2;
			kk += n - j;
			jx += incx;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* hpmv_impl — y ← α·A·x + β·y   (A Hermitian, packed)             */
/* ------------------------------------------------------------------ */
template <typename T>
void hpmv_impl(char uplo, int n, std::complex<T> alpha, const std::complex<T>* Ap,
               const std::complex<T>* x, int incx, std::complex<T> beta, std::complex<T>* y,
               int incy) {
	using C = std::complex<T>;
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0) {
		return;
	}

	// Scale y by beta
	{
		int iy = start_index(n, incy);
		if (beta == C(0)) {
			for (int i = 0; i < n; ++i) {
				y[iy] = C(0);
				iy += incy;
			}
		} else if (beta != C(1)) {
			for (int i = 0; i < n; ++i) {
				y[iy] *= beta;
				iy += incy;
			}
		}
	}

	if (alpha == C(0)) {
		return;
	}

	int kk = 0;
	if (ul == 'U') {
		int jx = start_index(n, incx);
		int jy = start_index(n, incy);
		for (int j = 0; j < n; ++j) {
			C temp1 = alpha * x[jx];
			C temp2 = C(0);
			int ix = start_index(n, incx);
			int iy = start_index(n, incy);
			for (int i = 0; i < j; ++i) {
				y[iy] += temp1 * Ap[kk + i];
				temp2 += std::conj(Ap[kk + i]) * x[ix];
				ix += incx;
				iy += incy;
			}
			y[jy] += temp1 * T(Ap[kk + j].real()) + alpha * temp2;
			kk += j + 1;
			jx += incx;
			jy += incy;
		}
	} else {
		int jx = start_index(n, incx);
		int jy = start_index(n, incy);
		for (int j = 0; j < n; ++j) {
			C temp1 = alpha * x[jx];
			C temp2 = C(0);
			y[jy] += temp1 * T(Ap[kk].real());
			int ix = jx;
			int iy = jy;
			for (int i = j + 1; i < n; ++i) {
				ix += incx;
				iy += incy;
				y[iy] += temp1 * Ap[kk + i - j];
				temp2 += std::conj(Ap[kk + i - j]) * x[ix];
			}
			y[jy] += alpha * temp2;
			kk += n - j;
			jx += incx;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* tpmv_impl — x ← op(A)·x   (A triangular, packed)                 */
/* ------------------------------------------------------------------ */
template <typename T>
void tpmv_impl(char uplo, char trans, char diag, int n, const T* Ap, T* x, int incx) {
	char ul = to_upper(uplo);
	char tr = to_upper(trans);
	char dg = to_upper(diag);
	if (!valid_uplo(uplo) || !valid_trans(trans) || !valid_diag(diag) || n <= 0 || incx == 0) {
		return;
	}
	bool unit = (dg == 'U');
	bool conj = (tr == 'C');

	if (tr == 'N') {
		if (ul == 'U') {
			int kk = 0;
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				if (x[jx] != T(0)) {
					T temp = x[jx];
					int ix = start_index(n, incx);
					for (int i = 0; i < j; ++i) {
						x[ix] += temp * Ap[kk + i];
						ix += incx;
					}
					if (!unit) {
						x[jx] *= Ap[kk + j];
					}
				}
				kk += j + 1;
				jx += incx;
			}
		} else {
			int kk = n * (n + 1) / 2 - 1;
			int jx = start_index(n, incx) + (n - 1) * incx;
			for (int j = n - 1; j >= 0; --j) {
				if (x[jx] != T(0)) {
					T temp = x[jx];
					int ix = start_index(n, incx) + (n - 1) * incx;
					for (int i = n - 1; i > j; --i) {
						x[ix] += temp * Ap[kk - (n - 1 - i)];
						ix -= incx;
					}
					if (!unit) {
						x[jx] *= Ap[kk - (n - 1 - j)];
					}
				}
				kk -= n - j;
				jx -= incx;
			}
		}
	} else {
		// Transpose or conjugate-transpose
		if (ul == 'U') {
			int kk = n * (n + 1) / 2 - 1;
			int jx = start_index(n, incx) + (n - 1) * incx;
			for (int j = n - 1; j >= 0; --j) {
				T temp = x[jx];
				if (!unit) {
					T aval = conj ? conj_val(Ap[kk]) : Ap[kk];
					temp *= aval;
				}
				int ix = jx;
				int k = kk - 1;
				for (int i = j - 1; i >= 0; --i) {
					ix -= incx;
					T aval = conj ? conj_val(Ap[k]) : Ap[k];
					temp += aval * x[ix];
					--k;
				}
				x[jx] = temp;
				kk -= j + 1;
				jx -= incx;
			}
		} else {
			int kk = 0;
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				T temp = x[jx];
				if (!unit) {
					T aval = conj ? conj_val(Ap[kk]) : Ap[kk];
					temp *= aval;
				}
				int ix = jx;
				int k = kk + 1;
				for (int i = j + 1; i < n; ++i) {
					ix += incx;
					T aval = conj ? conj_val(Ap[k]) : Ap[k];
					temp += aval * x[ix];
					++k;
				}
				x[jx] = temp;
				kk += n - j;
				jx += incx;
			}
		}
	}
}

/* ------------------------------------------------------------------ */
/* tpsv_impl — solve op(A)·x = b   (A triangular, packed)            */
/* ------------------------------------------------------------------ */
template <typename T>
void tpsv_impl(char uplo, char trans, char diag, int n, const T* Ap, T* x, int incx) {
	char ul = to_upper(uplo);
	char tr = to_upper(trans);
	char dg = to_upper(diag);
	if (!valid_uplo(uplo) || !valid_trans(trans) || !valid_diag(diag) || n <= 0 || incx == 0) {
		return;
	}
	bool unit = (dg == 'U');
	bool conj = (tr == 'C');

	if (tr == 'N') {
		if (ul == 'U') {
			int kk = n * (n + 1) / 2 - 1;
			int jx = start_index(n, incx) + (n - 1) * incx;
			for (int j = n - 1; j >= 0; --j) {
				if (!unit) {
					x[jx] /= Ap[kk];
				}
				T temp = x[jx];
				int ix = jx;
				int k = kk - 1;
				for (int i = j - 1; i >= 0; --i) {
					ix -= incx;
					x[ix] -= temp * Ap[k];
					--k;
				}
				kk -= j + 1;
				jx -= incx;
			}
		} else {
			int kk = 0;
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				if (!unit) {
					x[jx] /= Ap[kk];
				}
				T temp = x[jx];
				int ix = jx;
				int k = kk + 1;
				for (int i = j + 1; i < n; ++i) {
					ix += incx;
					x[ix] -= temp * Ap[k];
					++k;
				}
				kk += n - j;
				jx += incx;
			}
		}
	} else {
		if (ul == 'U') {
			int kk = 0;
			int jx = start_index(n, incx);
			for (int j = 0; j < n; ++j) {
				T temp = x[jx];
				int ix = start_index(n, incx);
				int k = kk;
				for (int i = 0; i < j; ++i) {
					T aval = conj ? conj_val(Ap[k]) : Ap[k];
					temp -= aval * x[ix];
					ix += incx;
					++k;
				}
				if (!unit) {
					T aval = conj ? conj_val(Ap[k]) : Ap[k];
					temp /= aval;
				}
				x[jx] = temp;
				kk += j + 1;
				jx += incx;
			}
		} else {
			int kk = n * (n + 1) / 2 - 1;
			int jx = start_index(n, incx) + (n - 1) * incx;
			for (int j = n - 1; j >= 0; --j) {
				T temp = x[jx];
				int ix = start_index(n, incx) + (n - 1) * incx;
				int k = kk;
				for (int i = n - 1; i > j; --i) {
					T aval = conj ? conj_val(Ap[k]) : Ap[k];
					temp -= aval * x[ix];
					ix -= incx;
					--k;
				}
				if (!unit) {
					T aval = conj ? conj_val(Ap[k]) : Ap[k];
					temp /= aval;
				}
				x[jx] = temp;
				kk -= n - j;
				jx -= incx;
			}
		}
	}
}

/* ------------------------------------------------------------------ */
/* spr_impl — A ← α·x·xᵀ + A   (A symmetric, packed)               */
/* ------------------------------------------------------------------ */
template <typename T>
void spr_impl(char uplo, int n, T alpha, const T* x, int incx, T* Ap) {
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || incx == 0) {
		return;
	}
	if (alpha == T(0)) {
		return;
	}

	int kk = 0;
	int jx = start_index(n, incx);
	if (ul == 'U') {
		for (int j = 0; j < n; ++j) {
			T temp = alpha * x[jx];
			int ix = start_index(n, incx);
			for (int i = 0; i <= j; ++i) {
				Ap[kk + i] += x[ix] * temp;
				ix += incx;
			}
			kk += j + 1;
			jx += incx;
		}
	} else {
		for (int j = 0; j < n; ++j) {
			T temp = alpha * x[jx];
			int ix = jx;
			for (int i = j; i < n; ++i) {
				Ap[kk + i - j] += x[ix] * temp;
				ix += incx;
			}
			kk += n - j;
			jx += incx;
		}
	}
}

/* ------------------------------------------------------------------ */
/* hpr_impl — A ← α·x·xᴴ + A   (A Hermitian, packed, α real)       */
/* ------------------------------------------------------------------ */
template <typename T>
void hpr_impl(char uplo, int n, T alpha, const std::complex<T>* x, int incx,
              std::complex<T>* Ap) {
	using C = std::complex<T>;
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || incx == 0) {
		return;
	}
	if (alpha == T(0)) {
		return;
	}

	int kk = 0;
	int jx = start_index(n, incx);
	if (ul == 'U') {
		for (int j = 0; j < n; ++j) {
			C temp = C(alpha) * std::conj(x[jx]);
			int ix = start_index(n, incx);
			for (int i = 0; i < j; ++i) {
				Ap[kk + i] += x[ix] * temp;
				ix += incx;
			}
			Ap[kk + j] = C(Ap[kk + j].real() + (x[jx] * temp).real());
			kk += j + 1;
			jx += incx;
		}
	} else {
		for (int j = 0; j < n; ++j) {
			C temp = C(alpha) * std::conj(x[jx]);
			Ap[kk] = C(Ap[kk].real() + (x[jx] * temp).real());
			int ix = jx;
			for (int i = j + 1; i < n; ++i) {
				ix += incx;
				Ap[kk + i - j] += x[ix] * temp;
			}
			kk += n - j;
			jx += incx;
		}
	}
}

/* ------------------------------------------------------------------ */
/* spr2_impl — A ← α·x·yᵀ + α·y·xᵀ + A   (symmetric packed)       */
/* ------------------------------------------------------------------ */
template <typename T>
void spr2_impl(char uplo, int n, T alpha, const T* x, int incx, const T* y, int incy, T* Ap) {
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0) {
		return;
	}
	if (alpha == T(0)) {
		return;
	}

	int kk = 0;
	int jx = start_index(n, incx);
	int jy = start_index(n, incy);
	if (ul == 'U') {
		for (int j = 0; j < n; ++j) {
			T temp1 = alpha * y[jy];
			T temp2 = alpha * x[jx];
			int ix = start_index(n, incx);
			int iy = start_index(n, incy);
			for (int i = 0; i <= j; ++i) {
				Ap[kk + i] += x[ix] * temp1 + y[iy] * temp2;
				ix += incx;
				iy += incy;
			}
			kk += j + 1;
			jx += incx;
			jy += incy;
		}
	} else {
		for (int j = 0; j < n; ++j) {
			T temp1 = alpha * y[jy];
			T temp2 = alpha * x[jx];
			int ix = jx;
			int iy = jy;
			for (int i = j; i < n; ++i) {
				Ap[kk + i - j] += x[ix] * temp1 + y[iy] * temp2;
				ix += incx;
				iy += incy;
			}
			kk += n - j;
			jx += incx;
			jy += incy;
		}
	}
}

/* ------------------------------------------------------------------ */
/* hpr2_impl — A ← α·x·yᴴ + conj(α)·y·xᴴ + A   (Hermitian packed) */
/* ------------------------------------------------------------------ */
template <typename T>
void hpr2_impl(char uplo, int n, std::complex<T> alpha, const std::complex<T>* x, int incx,
               const std::complex<T>* y, int incy, std::complex<T>* Ap) {
	using C = std::complex<T>;
	char ul = to_upper(uplo);
	if (!valid_uplo(uplo) || n <= 0 || incx == 0 || incy == 0) {
		return;
	}
	if (alpha == C(0)) {
		return;
	}

	int kk = 0;
	int jx = start_index(n, incx);
	int jy = start_index(n, incy);
	if (ul == 'U') {
		for (int j = 0; j < n; ++j) {
			C temp1 = alpha * std::conj(y[jy]);
			C temp2 = std::conj(alpha * x[jx]);
			int ix = start_index(n, incx);
			int iy = start_index(n, incy);
			for (int i = 0; i < j; ++i) {
				Ap[kk + i] += x[ix] * temp1 + y[iy] * temp2;
				ix += incx;
				iy += incy;
			}
			Ap[kk + j] = C((Ap[kk + j] + x[jx] * temp1 + y[jy] * temp2).real());
			kk += j + 1;
			jx += incx;
			jy += incy;
		}
	} else {
		for (int j = 0; j < n; ++j) {
			C temp1 = alpha * std::conj(y[jy]);
			C temp2 = std::conj(alpha * x[jx]);
			Ap[kk] = C((Ap[kk] + x[jx] * temp1 + y[jy] * temp2).real());
			int ix = jx;
			int iy = jy;
			for (int i = j + 1; i < n; ++i) {
				ix += incx;
				iy += incy;
				Ap[kk + i - j] += x[ix] * temp1 + y[iy] * temp2;
			}
			kk += n - j;
			jx += incx;
			jy += incy;
		}
	}
}

} // anonymous namespace

/* ====================================================================
 * Level 2 Public API wrappers
 * ==================================================================== */

/* gemv */
void sgemv(char trans, int m, int n, float alpha, const float* A, int lda, const float* x,
           int incx, float beta, float* y, int incy) {
	gemv_impl(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
void dgemv(char trans, int m, int n, double alpha, const double* A, int lda, const double* x,
           int incx, double beta, double* y, int incy) {
	gemv_impl(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
void cgemv(char trans, int m, int n, std::complex<float> alpha, const std::complex<float>* A,
           int lda, const std::complex<float>* x, int incx, std::complex<float> beta,
           std::complex<float>* y, int incy) {
	gemv_impl(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
void zgemv(char trans, int m, int n, std::complex<double> alpha, const std::complex<double>* A,
           int lda, const std::complex<double>* x, int incx, std::complex<double> beta,
           std::complex<double>* y, int incy) {
	gemv_impl(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

/* symv */
void ssymv(char uplo, int n, float alpha, const float* A, int lda, const float* x, int incx,
           float beta, float* y, int incy) {
	symv_impl(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}
void dsymv(char uplo, int n, double alpha, const double* A, int lda, const double* x, int incx,
           double beta, double* y, int incy) {
	symv_impl(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

/* hemv */
void chemv(char uplo, int n, std::complex<float> alpha, const std::complex<float>* A, int lda,
           const std::complex<float>* x, int incx, std::complex<float> beta,
           std::complex<float>* y, int incy) {
	hemv_impl(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}
void zhemv(char uplo, int n, std::complex<double> alpha, const std::complex<double>* A, int lda,
           const std::complex<double>* x, int incx, std::complex<double> beta,
           std::complex<double>* y, int incy) {
	hemv_impl(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

/* trmv */
void strmv(char uplo, char trans, char diag, int n, const float* A, int lda, float* x,
           int incx) {
	trmv_impl(uplo, trans, diag, n, A, lda, x, incx);
}
void dtrmv(char uplo, char trans, char diag, int n, const double* A, int lda, double* x,
           int incx) {
	trmv_impl(uplo, trans, diag, n, A, lda, x, incx);
}
void ctrmv(char uplo, char trans, char diag, int n, const std::complex<float>* A, int lda,
           std::complex<float>* x, int incx) {
	trmv_impl(uplo, trans, diag, n, A, lda, x, incx);
}
void ztrmv(char uplo, char trans, char diag, int n, const std::complex<double>* A, int lda,
           std::complex<double>* x, int incx) {
	trmv_impl(uplo, trans, diag, n, A, lda, x, incx);
}

/* trsv */
void strsv(char uplo, char trans, char diag, int n, const float* A, int lda, float* x,
           int incx) {
	trsv_impl(uplo, trans, diag, n, A, lda, x, incx);
}
void dtrsv(char uplo, char trans, char diag, int n, const double* A, int lda, double* x,
           int incx) {
	trsv_impl(uplo, trans, diag, n, A, lda, x, incx);
}
void ctrsv(char uplo, char trans, char diag, int n, const std::complex<float>* A, int lda,
           std::complex<float>* x, int incx) {
	trsv_impl(uplo, trans, diag, n, A, lda, x, incx);
}
void ztrsv(char uplo, char trans, char diag, int n, const std::complex<double>* A, int lda,
           std::complex<double>* x, int incx) {
	trsv_impl(uplo, trans, diag, n, A, lda, x, incx);
}

/* ger */
void sger(int m, int n, float alpha, const float* x, int incx, const float* y, int incy,
          float* A, int lda) {
	ger_impl(m, n, alpha, x, incx, y, incy, A, lda);
}
void dger(int m, int n, double alpha, const double* x, int incx, const double* y, int incy,
          double* A, int lda) {
	ger_impl(m, n, alpha, x, incx, y, incy, A, lda);
}

/* geru / gerc */
void cgeru(int m, int n, std::complex<float> alpha, const std::complex<float>* x, int incx,
           const std::complex<float>* y, int incy, std::complex<float>* A, int lda) {
	geru_impl(m, n, alpha, x, incx, y, incy, A, lda);
}
void zgeru(int m, int n, std::complex<double> alpha, const std::complex<double>* x, int incx,
           const std::complex<double>* y, int incy, std::complex<double>* A, int lda) {
	geru_impl(m, n, alpha, x, incx, y, incy, A, lda);
}
void cgerc(int m, int n, std::complex<float> alpha, const std::complex<float>* x, int incx,
           const std::complex<float>* y, int incy, std::complex<float>* A, int lda) {
	gerc_impl(m, n, alpha, x, incx, y, incy, A, lda);
}
void zgerc(int m, int n, std::complex<double> alpha, const std::complex<double>* x, int incx,
           const std::complex<double>* y, int incy, std::complex<double>* A, int lda) {
	gerc_impl(m, n, alpha, x, incx, y, incy, A, lda);
}

/* syr */
void ssyr(char uplo, int n, float alpha, const float* x, int incx, float* A, int lda) {
	syr_impl(uplo, n, alpha, x, incx, A, lda);
}
void dsyr(char uplo, int n, double alpha, const double* x, int incx, double* A, int lda) {
	syr_impl(uplo, n, alpha, x, incx, A, lda);
}

/* her */
void cher(char uplo, int n, float alpha, const std::complex<float>* x, int incx,
          std::complex<float>* A, int lda) {
	her_impl(uplo, n, alpha, x, incx, A, lda);
}
void zher(char uplo, int n, double alpha, const std::complex<double>* x, int incx,
          std::complex<double>* A, int lda) {
	her_impl(uplo, n, alpha, x, incx, A, lda);
}

/* syr2 */
void ssyr2(char uplo, int n, float alpha, const float* x, int incx, const float* y, int incy,
           float* A, int lda) {
	syr2_impl(uplo, n, alpha, x, incx, y, incy, A, lda);
}
void dsyr2(char uplo, int n, double alpha, const double* x, int incx, const double* y, int incy,
           double* A, int lda) {
	syr2_impl(uplo, n, alpha, x, incx, y, incy, A, lda);
}

/* her2 */
void cher2(char uplo, int n, std::complex<float> alpha, const std::complex<float>* x, int incx,
           const std::complex<float>* y, int incy, std::complex<float>* A, int lda) {
	her2_impl(uplo, n, alpha, x, incx, y, incy, A, lda);
}
void zher2(char uplo, int n, std::complex<double> alpha, const std::complex<double>* x, int incx,
           const std::complex<double>* y, int incy, std::complex<double>* A, int lda) {
	her2_impl(uplo, n, alpha, x, incx, y, incy, A, lda);
}

/* gbmv */
void sgbmv(char trans, int m, int n, int kl, int ku, float alpha, const float* A, int lda,
           const float* x, int incx, float beta, float* y, int incy) {
	gbmv_impl(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}
void dgbmv(char trans, int m, int n, int kl, int ku, double alpha, const double* A, int lda,
           const double* x, int incx, double beta, double* y, int incy) {
	gbmv_impl(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}
void cgbmv(char trans, int m, int n, int kl, int ku, std::complex<float> alpha,
           const std::complex<float>* A, int lda, const std::complex<float>* x, int incx,
           std::complex<float> beta, std::complex<float>* y, int incy) {
	gbmv_impl(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}
void zgbmv(char trans, int m, int n, int kl, int ku, std::complex<double> alpha,
           const std::complex<double>* A, int lda, const std::complex<double>* x, int incx,
           std::complex<double> beta, std::complex<double>* y, int incy) {
	gbmv_impl(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

/* sbmv */
void ssbmv(char uplo, int n, int k, float alpha, const float* A, int lda, const float* x,
           int incx, float beta, float* y, int incy) {
	sbmv_impl(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}
void dsbmv(char uplo, int n, int k, double alpha, const double* A, int lda, const double* x,
           int incx, double beta, double* y, int incy) {
	sbmv_impl(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

/* hbmv */
void chbmv(char uplo, int n, int k, std::complex<float> alpha, const std::complex<float>* A,
           int lda, const std::complex<float>* x, int incx, std::complex<float> beta,
           std::complex<float>* y, int incy) {
	hbmv_impl(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}
void zhbmv(char uplo, int n, int k, std::complex<double> alpha, const std::complex<double>* A,
           int lda, const std::complex<double>* x, int incx, std::complex<double> beta,
           std::complex<double>* y, int incy) {
	hbmv_impl(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

/* tbmv */
void stbmv(char uplo, char trans, char diag, int n, int k, const float* A, int lda, float* x,
           int incx) {
	tbmv_impl(uplo, trans, diag, n, k, A, lda, x, incx);
}
void dtbmv(char uplo, char trans, char diag, int n, int k, const double* A, int lda, double* x,
           int incx) {
	tbmv_impl(uplo, trans, diag, n, k, A, lda, x, incx);
}
void ctbmv(char uplo, char trans, char diag, int n, int k, const std::complex<float>* A, int lda,
           std::complex<float>* x, int incx) {
	tbmv_impl(uplo, trans, diag, n, k, A, lda, x, incx);
}
void ztbmv(char uplo, char trans, char diag, int n, int k, const std::complex<double>* A,
           int lda, std::complex<double>* x, int incx) {
	tbmv_impl(uplo, trans, diag, n, k, A, lda, x, incx);
}

/* tbsv */
void stbsv(char uplo, char trans, char diag, int n, int k, const float* A, int lda, float* x,
           int incx) {
	tbsv_impl(uplo, trans, diag, n, k, A, lda, x, incx);
}
void dtbsv(char uplo, char trans, char diag, int n, int k, const double* A, int lda, double* x,
           int incx) {
	tbsv_impl(uplo, trans, diag, n, k, A, lda, x, incx);
}
void ctbsv(char uplo, char trans, char diag, int n, int k, const std::complex<float>* A, int lda,
           std::complex<float>* x, int incx) {
	tbsv_impl(uplo, trans, diag, n, k, A, lda, x, incx);
}
void ztbsv(char uplo, char trans, char diag, int n, int k, const std::complex<double>* A,
           int lda, std::complex<double>* x, int incx) {
	tbsv_impl(uplo, trans, diag, n, k, A, lda, x, incx);
}

/* spmv */
void sspmv(char uplo, int n, float alpha, const float* Ap, const float* x, int incx, float beta,
           float* y, int incy) {
	spmv_impl(uplo, n, alpha, Ap, x, incx, beta, y, incy);
}
void dspmv(char uplo, int n, double alpha, const double* Ap, const double* x, int incx,
           double beta, double* y, int incy) {
	spmv_impl(uplo, n, alpha, Ap, x, incx, beta, y, incy);
}

/* hpmv */
void chpmv(char uplo, int n, std::complex<float> alpha, const std::complex<float>* Ap,
           const std::complex<float>* x, int incx, std::complex<float> beta,
           std::complex<float>* y, int incy) {
	hpmv_impl(uplo, n, alpha, Ap, x, incx, beta, y, incy);
}
void zhpmv(char uplo, int n, std::complex<double> alpha, const std::complex<double>* Ap,
           const std::complex<double>* x, int incx, std::complex<double> beta,
           std::complex<double>* y, int incy) {
	hpmv_impl(uplo, n, alpha, Ap, x, incx, beta, y, incy);
}

/* tpmv */
void stpmv(char uplo, char trans, char diag, int n, const float* Ap, float* x, int incx) {
	tpmv_impl(uplo, trans, diag, n, Ap, x, incx);
}
void dtpmv(char uplo, char trans, char diag, int n, const double* Ap, double* x, int incx) {
	tpmv_impl(uplo, trans, diag, n, Ap, x, incx);
}
void ctpmv(char uplo, char trans, char diag, int n, const std::complex<float>* Ap,
           std::complex<float>* x, int incx) {
	tpmv_impl(uplo, trans, diag, n, Ap, x, incx);
}
void ztpmv(char uplo, char trans, char diag, int n, const std::complex<double>* Ap,
           std::complex<double>* x, int incx) {
	tpmv_impl(uplo, trans, diag, n, Ap, x, incx);
}

/* tpsv */
void stpsv(char uplo, char trans, char diag, int n, const float* Ap, float* x, int incx) {
	tpsv_impl(uplo, trans, diag, n, Ap, x, incx);
}
void dtpsv(char uplo, char trans, char diag, int n, const double* Ap, double* x, int incx) {
	tpsv_impl(uplo, trans, diag, n, Ap, x, incx);
}
void ctpsv(char uplo, char trans, char diag, int n, const std::complex<float>* Ap,
           std::complex<float>* x, int incx) {
	tpsv_impl(uplo, trans, diag, n, Ap, x, incx);
}
void ztpsv(char uplo, char trans, char diag, int n, const std::complex<double>* Ap,
           std::complex<double>* x, int incx) {
	tpsv_impl(uplo, trans, diag, n, Ap, x, incx);
}

/* spr */
void sspr(char uplo, int n, float alpha, const float* x, int incx, float* Ap) {
	spr_impl(uplo, n, alpha, x, incx, Ap);
}
void dspr(char uplo, int n, double alpha, const double* x, int incx, double* Ap) {
	spr_impl(uplo, n, alpha, x, incx, Ap);
}

/* hpr */
void chpr(char uplo, int n, float alpha, const std::complex<float>* x, int incx,
          std::complex<float>* Ap) {
	hpr_impl(uplo, n, alpha, x, incx, Ap);
}
void zhpr(char uplo, int n, double alpha, const std::complex<double>* x, int incx,
          std::complex<double>* Ap) {
	hpr_impl(uplo, n, alpha, x, incx, Ap);
}

/* spr2 */
void sspr2(char uplo, int n, float alpha, const float* x, int incx, const float* y, int incy,
           float* Ap) {
	spr2_impl(uplo, n, alpha, x, incx, y, incy, Ap);
}
void dspr2(char uplo, int n, double alpha, const double* x, int incx, const double* y, int incy,
           double* Ap) {
	spr2_impl(uplo, n, alpha, x, incx, y, incy, Ap);
}

/* hpr2 */
void chpr2(char uplo, int n, std::complex<float> alpha, const std::complex<float>* x, int incx,
           const std::complex<float>* y, int incy, std::complex<float>* Ap) {
	hpr2_impl(uplo, n, alpha, x, incx, y, incy, Ap);
}
void zhpr2(char uplo, int n, std::complex<double> alpha, const std::complex<double>* x, int incx,
           const std::complex<double>* y, int incy, std::complex<double>* Ap) {
	hpr2_impl(uplo, n, alpha, x, incx, y, incy, Ap);
}

} // namespace theblas
