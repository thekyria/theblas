#pragma once

#include <complex>

namespace theblas::detail {

inline int start_index(int n, int inc) {
	return (inc > 0) ? 0 : (1 - n) * inc;
}

inline char to_upper(char value) {
	return static_cast<char>(value >= 'a' && value <= 'z' ? value - 32 : value);
}

inline bool valid_trans(char value) {
	const char upper = to_upper(value);
	return upper == 'N' || upper == 'T' || upper == 'C';
}

inline bool valid_uplo(char value) {
	const char upper = to_upper(value);
	return upper == 'U' || upper == 'L';
}

inline bool valid_diag(char value) {
	const char upper = to_upper(value);
	return upper == 'U' || upper == 'N';
}

template <typename T>
T conj_val(T value) {
	return value;
}

template <typename T>
std::complex<T> conj_val(std::complex<T> value) {
	return std::conj(value);
}

} // namespace theblas::detail