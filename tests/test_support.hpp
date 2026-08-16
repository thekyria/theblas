#pragma once

#include <cmath>
#include <complex>

namespace theblas::test {

inline bool almost_equal(float lhs, float rhs, float epsilon = 1e-6F) {
    return std::fabs(lhs - rhs) <= epsilon;
}

inline bool almost_equal(double lhs, double rhs, double epsilon = 1e-12) {
    return std::fabs(lhs - rhs) <= epsilon;
}

inline bool almost_equal(std::complex<float> lhs, std::complex<float> rhs, float epsilon = 1e-6F) {
    return almost_equal(lhs.real(), rhs.real(), epsilon) && almost_equal(lhs.imag(), rhs.imag(), epsilon);
}

inline bool almost_equal(std::complex<double> lhs, std::complex<double> rhs, double epsilon = 1e-12) {
    return almost_equal(lhs.real(), rhs.real(), epsilon) && almost_equal(lhs.imag(), rhs.imag(), epsilon);
}

} // namespace theblas::test