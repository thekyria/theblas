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

### Build with CMake Presets

For CMake 3.21 and newer, the repository provides [CMakePresets.json](CMakePresets.json) with reusable configure, build, and test profiles.

List available presets:

```bash
cmake --list-presets
```

Common workflows:

```bash
cmake --preset gcc-debug
cmake --build --preset gcc-debug
ctest --preset gcc-debug
```

```bash
cmake --preset clang-debug-sanitized
cmake --build --preset clang-debug-sanitized
ctest --preset clang-debug-sanitized
```

```bash
cmake --preset clang-release
cmake --build --preset clang-release
ctest --preset clang-release
```

Available configure presets:

- `gcc-debug`
- `clang-debug-sanitized`
- `clang-release`
- `msvc-release`

The `msvc-release` preset is intended for Windows in a Visual Studio Developer Command Prompt.

### Build Manually

```bash
mkdir build && cd build
cmake ..
cmake --build .
ctest
```

### Build Options (Compiler Flags)

Configurable CMake options:

- `THEBLAS_ENABLE_STRICT_WARNINGS=ON|OFF` (default: `ON`)
- `THEBLAS_WARNINGS_AS_ERRORS=ON|OFF` (default: `OFF`)
- `THEBLAS_ENABLE_SANITIZERS=ON|OFF` (default: `OFF`, Debug only, Clang/GCC)

Example:

```bash
cmake -S . -B build \
	-DCMAKE_BUILD_TYPE=Debug \
	-DTHEBLAS_ENABLE_STRICT_WARNINGS=ON \
	-DTHEBLAS_WARNINGS_AS_ERRORS=ON \
	-DTHEBLAS_ENABLE_SANITIZERS=ON
cmake --build build
ctest --test-dir build
```

### Toolchain Files

This repository includes toolchain files for Clang and MSVC:

- `cmake/toolchains/clang.cmake`
- `cmake/toolchains/msvc.cmake`

Clang example:

```bash
cmake -S . -B build-clang \
	-DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/clang.cmake \
	-DCMAKE_BUILD_TYPE=Release
cmake --build build-clang
ctest --test-dir build-clang
```

MSVC example (run on Windows in a Visual Studio Developer Command Prompt):

```powershell
cmake -S . -B build-msvc -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/msvc.cmake
cmake --build build-msvc --config Release
ctest --test-dir build-msvc -C Release
```

## Build with Makefile

This repository also provides a top-level `Makefile` with common development targets.

```bash
make help
make build
make test
make install
make docs
make clean
```

Useful variables:

- `BUILD_DIR` (default: `build`)
- `CONFIG` (default: `Release`)
- `PREFIX` (default: `./install`)

Example:

```bash
make test CONFIG=Debug BUILD_DIR=build-debug
```

## Dev Container

This repository includes a VS Code dev container in `.devcontainer/`.

## GitHub Codespaces

This repository is ready for GitHub Codespaces using the same `.devcontainer/` configuration.

To launch:

1. Open the repository on GitHub.
2. Click `Code` -> `Codespaces` -> `Create codespace on main`.
3. Wait for the environment to build.

After creation, the container runs:

```bash
cmake -S . -B build && cmake --build build && ctest --test-dir build
```

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
