"""Call theblas through its C ABI from Python with ctypes (no build step on the Python side)."""

import ctypes
import sys

lib = ctypes.CDLL(sys.argv[1])

c_int, c_char, c_double = ctypes.c_int, ctypes.c_char, ctypes.c_double
p_double = ctypes.POINTER(c_double)

lib.theblas_dgemm.restype = None
lib.theblas_dgemm.argtypes = [c_char, c_char, c_int, c_int, c_int, c_double, p_double, c_int,
                              p_double, c_int, c_double, p_double, c_int]
lib.theblas_zdotu.restype = None
lib.theblas_zdotu.argtypes = [c_int, p_double, c_int, p_double, c_int, p_double]


def array(values):
    return (c_double * len(values))(*values)


# C = A * B with A (2x3) and B (3x2) in column-major order.
a = array([1, 4, 2, 5, 3, 6])
b = array([7, 9, 11, 8, 10, 12])
c = array([0] * 4)
lib.theblas_dgemm(b"N", b"N", 2, 2, 3, 1.0, a, 2, b, 3, 0.0, c, 2)
assert list(c) == [58, 139, 64, 154], list(c)

# Complex vectors are interleaved {re, im} pairs: x = [1+2i, 3-i], y = [i, 2].
x = array([1, 2, 3, -1])
y = array([0, 1, 2, 0])
result = array([0, 0])
lib.theblas_zdotu(2, x, 1, y, 1, result)
assert complex(*result) == (1 + 2j) * 1j + (3 - 1j) * 2, list(result)

print("theblas_c ctypes test passed")
