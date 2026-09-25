#pragma once

#include <complex>
#include <cstddef>

namespace theblas::detail {

using index_t = std::ptrdiff_t;

inline index_t offset(int i, int inc) {
    return static_cast<index_t>(i) * static_cast<index_t>(inc);
}

inline index_t idx(int i, int j, int ld) {
    return static_cast<index_t>(i) + static_cast<index_t>(j) * static_cast<index_t>(ld);
}

inline index_t packed_size(int n) {
    return static_cast<index_t>(n) * static_cast<index_t>(n + 1) / 2;
}

inline index_t start_index(int n, int inc) {
    return (inc > 0) ? 0 : offset(1 - n, inc);
}

error_handler_t current_error_handler();

inline void report_error(const char *routine, int param) {
    if (const auto handler = current_error_handler(); handler != nullptr) {
        handler(routine, param);
    }
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

template <typename T> T conj_val(T value) {
    return value;
}

template <typename T> std::complex<T> conj_val(std::complex<T> value) {
    return std::conj(value);
}

} // namespace theblas::detail