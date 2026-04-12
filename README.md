# theblas

A minimal C++17 BLAS-style library scaffold with a simple Level 1 BLAS implementation.

## Implemented Level 1 BLAS Routines

Implemented with BLAS-like naming and increments (`incx`, `incy`):

- Real single precision (`s*`) and real double precision (`d*`)
- Complex single precision (`c*`) and complex double precision (`z*`)

### Vector Operations

- `swap`: `sswap`, `dswap`
- `copy`: `scopy`, `dcopy`
- `axpy`: `saxpy`, `daxpy`
- `scal`: `sscal`, `dscal`
- `dot`: `sdot`, `ddot`
- `nrm2`: `snrm2`, `dnrm2`
- `asum`: `sasum`, `dasum`
- `iamax`: `isamax`, `idamax` (returns 1-based index to match Netlib BLAS semantics)

Complex variants:

- `swap`: `cswap`, `zswap`
- `copy`: `ccopy`, `zcopy`
- `axpy`: `caxpy`, `zaxpy`
- `scal`: `cscal`, `zscal`, `csscal`, `zdscal`
- `dot`: `cdotu`, `cdotc`, `zdotu`, `zdotc`
- `nrm2`: `scnrm2`, `dznrm2`
- `asum`: `scasum`, `dzasum`
- `iamax`: `icamax`, `izamax` (1-based)

### Rotations

- Plane rotation apply: `srot`, `drot`, `csrot`, `zdrot`
- Givens parameter generation: `srotg`, `drotg`, `crotg`, `zrotg`
- Modified Givens apply/generation: `srotm`, `drotm`, `srotmg`, `drotmg`

`srotmg` and `drotmg` are intentionally simple reference implementations suitable for learning and testing.

## Netlib Level 1 Compatibility Table

| Netlib BLAS Routine Family | Implemented in theblas |
| --- | --- |
| `?swap` | `sswap`, `dswap`, `cswap`, `zswap` |
| `?copy` | `scopy`, `dcopy`, `ccopy`, `zcopy` |
| `?axpy` | `saxpy`, `daxpy`, `caxpy`, `zaxpy` |
| `?scal` | `sscal`, `dscal`, `cscal`, `zscal`, `csscal`, `zdscal` |
| `?dot` | `sdot`, `ddot`, `cdotu`, `cdotc`, `zdotu`, `zdotc` |
| `?nrm2` | `snrm2`, `dnrm2`, `scnrm2`, `dznrm2` |
| `?asum` | `sasum`, `dasum`, `scasum`, `dzasum` |
| `i?amax` | `isamax`, `idamax`, `icamax`, `izamax` |
| `?rot` | `srot`, `drot`, `csrot`, `zdrot` |
| `?rotg` | `srotg`, `drotg`, `crotg`, `zrotg` |
| `?rotm` | `srotm`, `drotm` |
| `?rotmg` | `srotmg`, `drotmg` |

Notes:

- `?` follows Netlib precision/type conventions (`s`, `d`, `c`, `z`).
- `cdotu`/`zdotu` are unconjugated complex dot products; `cdotc`/`zdotc` are conjugated.
- `*amax` routines return 1-based indices to match Netlib semantics.

## Example

```cpp
#include "theblas/theblas.h"

float x[] = {1.0f, 2.0f, 3.0f};
float y[] = {10.0f, 20.0f, 30.0f};

theblas::saxpy(3, 0.5f, x, 1, y, 1);
// y = [10.5, 21.0, 31.5]
```

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
ctest
```

## Dev Container

This repository includes a VS Code dev container in `.devcontainer/`.

To use it:

1. Install Docker and the VS Code Dev Containers extension.
2. Open the repository in VS Code.
3. Run `Dev Containers: Reopen in Container`.

The container provisions C++/CMake tooling plus Doxygen/Graphviz, then runs:

```bash
cmake -S . -B build && cmake --build build && ctest --test-dir build
```

## Use with CPM.cmake

This project is CPM-friendly and exposes a namespaced target: `theblas::theblas`.

```cmake
include(cmake/CPM.cmake)

CPMAddPackage(
	NAME theblas
	GITHUB_REPOSITORY thekyria_github/theblas
	GIT_TAG main
)

target_link_libraries(your_target PRIVATE theblas::theblas)
```

Notes:

- As a dependency, tests are disabled by default.
- To force tests on in a superproject, set `THEBLAS_BUILD_TESTS=ON`.

## Use with find_package (CONFIG)

After installing the project, consumers can use CMake package config mode:

```cmake
find_package(theblas CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE theblas::theblas)
```

You can install locally with:

```bash
cmake -S . -B build
cmake --build build
cmake --install build --prefix ./install
```

## Generate API Docs (Doxygen)

Generate HTML API documentation from the Doxygen comments in `include/theblas/theblas.h`.

### 1. Install Doxygen

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y doxygen graphviz
```

### 2. Create a Doxygen configuration file

From the repository root:

```bash
doxygen -g Doxyfile
```

### 3. Configure Doxygen

Edit `Doxyfile` and set at least these options:

```txt
PROJECT_NAME           = "theblas"
INPUT                  = include src README.md
FILE_PATTERNS          = *.h *.hpp *.cpp
RECURSIVE              = YES
EXTRACT_ALL            = YES
GENERATE_HTML          = YES
GENERATE_LATEX         = NO
OUTPUT_DIRECTORY       = docs/doxygen
WARN_IF_UNDOCUMENTED   = YES
```

Optional call graph support:

```txt
HAVE_DOT               = YES
CALL_GRAPH             = YES
CALLER_GRAPH           = YES
```

### 4. Generate documentation

```bash
doxygen Doxyfile
```

### 5. Open the generated docs

Main page:

- `docs/doxygen/html/index.html`
