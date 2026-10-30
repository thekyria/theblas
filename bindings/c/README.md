# theblas C ABI

`theblas_c` is an optional shared library that exposes every theblas routine through a plain C
interface. It exists so that C code and foreign-function interfaces (Python `ctypes`/`cffi`,
Julia `ccall`, Rust FFI, ...) can call theblas. It is a separate CMake project and is not part
of the root build.

## Build

From the repository root:

```bash
cmake -S bindings/c -B build-c
cmake --build build-c
ctest --test-dir build-c
```

By default theblas is compiled from this source tree. To link an installed theblas package
instead, pass `-DTHEBLAS_C_FROM_SOURCE=OFF` and point `CMAKE_PREFIX_PATH` at it. That package
must have been built with `-DCMAKE_POSITION_INDEPENDENT_CODE=ON` (or as a shared library) so it
can be linked into `theblas_c`.

## Conventions

The header is [`include/theblas/theblas_c.h`](include/theblas/theblas_c.h).

- Each C++ routine `theblas::xyz` is available as `theblas_xyz`, with the same argument order
  and semantics: column-major storage, BLAS strides, 1-based `i?amax` results.
- The `theblas_` prefix avoids symbol clashes with any CBLAS or Fortran BLAS loaded in the same
  process (for example, the OpenBLAS that NumPy loads).
- Complex numbers are interleaved `{real, imag}` pairs:
  - complex arrays are passed as `float *` / `double *`;
  - complex scalars (such as `alpha`) are passed as a pointer to one pair;
  - complex results (`?dotu`, `?dotc`) are written to a trailing `result` pair.
- `theblas_set_error_handler` installs the same process-global Level-2/3 argument handler as
  the C++ API.

## Python example

```python
import ctypes

lib = ctypes.CDLL("build-c/libtheblas_c.so")
p_double = ctypes.POINTER(ctypes.c_double)
lib.theblas_daxpy.restype = None
lib.theblas_daxpy.argtypes = [ctypes.c_int, ctypes.c_double, p_double, ctypes.c_int,
                              p_double, ctypes.c_int]

x = (ctypes.c_double * 3)(1, 2, 3)
y = (ctypes.c_double * 3)(10, 20, 30)
lib.theblas_daxpy(3, 0.5, x, 1, y, 1)
print(list(y))  # [10.5, 21.0, 31.5]
```

[`tests/ctypes_smoke.py`](tests/ctypes_smoke.py) shows complex arguments as well.

## Tests

- `theblas_c_smoke`: a C99 program, which also checks that the header is valid C.
- `theblas_c_coverage`: fails if a routine in `theblas.h` has no `theblas_` wrapper.
- `theblas_c_ctypes`: calls the library from Python through `ctypes`.

The two Python tests run only when CMake finds a Python 3 interpreter.
