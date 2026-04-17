# theblas

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12538/badge)](https://www.bestpractices.dev/projects/12538)
[![Build](https://github.com/thekyria/theblas/actions/workflows/build.yml/badge.svg)](https://github.com/thekyria/theblas/actions/workflows/build.yml)
[![MSVC](https://github.com/thekyria/theblas/actions/workflows/msvc.yml/badge.svg)](https://github.com/thekyria/theblas/actions/workflows/msvc.yml)
[![Coverage](https://github.com/thekyria/theblas/actions/workflows/coverage.yml/badge.svg)](https://github.com/thekyria/theblas/actions/workflows/coverage.yml)
[![Coverage %](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/thekyria/8e1037ada9fe1e598aefebfe56b920ca/raw/theblas-coverage.json)](https://github.com/thekyria/theblas/actions/workflows/coverage.yml)
[![SonarCloud](https://sonarcloud.io/api/project_badges/measure?project=thekyria_theblas&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=thekyria_theblas)
[![CodeQL](https://github.com/thekyria/theblas/actions/workflows/codeql.yml/badge.svg)](https://github.com/thekyria/theblas/actions/workflows/codeql.yml)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-3.15%2B-blue.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A minimal, header-clean C++17 library implementing all [Netlib BLAS Level 1](https://netlib.org/blas/) routines for real and complex vectors — including rotations — with full BLAS stride semantics, a CMake build system, vcpkg and Conan 2.x packaging, and automated documentation publishing.

**[Documentation site](https://thekyria.github.io/theblas/)** — API reference and mathematical reference.

## Features

- Complete Level 1 BLAS coverage: `swap`, `copy`, `axpy`, `scal`, `dot`, `nrm2`, `asum`, `iamax`, `rot`, `rotg`, `rotm`, `rotmg`
- All four precision variants: `s` (float), `d` (double), `c` (complex float), `z` (complex double)
- BLAS-compatible stride parameters (`incx`, `incy`), negative strides, and 1-based `iamax` index returns
- Conjugated (`cdotc`/`zdotc`) and unconjugated (`cdotu`/`zdotu`) complex dot products
- Mixed-precision real-scalar scaling for complex vectors (`csscal`, `zdscal`)
- Single header, no dependencies beyond the C++17 standard library
- Cross-platform: GCC, Clang, MSVC; native and cross-compiled ARM Linux / Cortex-M targets
- vcpkg git registry and Conan 2.x recipe included

## Quick start

```cpp
#include "theblas/theblas.h"

float x[] = {1.0F, 2.0F, 3.0F};
float y[] = {10.0F, 20.0F, 30.0F};

// y ← 0.5·x + y
theblas::saxpy(3, 0.5F, x, 1, y, 1);
// y = [10.5, 21.0, 31.5]

// Euclidean norm
float n = theblas::snrm2(3, x, 1);  // √(1+4+9) ≈ 3.742

// 1-based index of the largest element
int k = theblas::isamax(3, x, 1);   // 3  (x[2] = 3.0)
```

## Routines

| Routine family | Real variants | Complex variants |
| --- | --- | --- |
| `?swap` | `sswap`, `dswap` | `cswap`, `zswap` |
| `?copy` | `scopy`, `dcopy` | `ccopy`, `zcopy` |
| `?axpy` | `saxpy`, `daxpy` | `caxpy`, `zaxpy` |
| `?scal` | `sscal`, `dscal` | `cscal`, `zscal`, `csscal`*, `zdscal`* |
| `?dot` | `sdot`, `ddot` | `cdotu`†, `cdotc`‡, `zdotu`†, `zdotc`‡ |
| `?nrm2` | `snrm2`, `dnrm2` | `scnrm2`, `dznrm2` |
| `?asum` | `sasum`, `dasum` | `scasum`, `dzasum` |
| `i?amax` | `isamax`, `idamax` | `icamax`, `izamax` |
| `?rot` | `srot`, `drot` | `csrot`, `zdrot` |
| `?rotg` | `srotg`, `drotg` | `crotg`, `zrotg` |
| `?rotm` | `srotm`, `drotm` | — |
| `?rotmg` | `srotmg`, `drotmg` | — |

\* real scalar applied to complex vector — \
† unconjugated complex dot product — ‡ conjugated complex dot product

`i?amax` returns a **1-based** index to match Netlib BLAS semantics.

For the mathematical definitions of every operation see the
[Mathematical Reference](https://thekyria.github.io/theblas/math-reference.html).

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

## Examples

The `examples/` folder contains standalone programs you can build and run:

| File | Routines demonstrated |
| --- | --- |
| [`examples/vector_ops.cpp`](examples/vector_ops.cpp) | `snrm2`, `sscal`, `saxpy`, `sdot`, `isamax` |
| [`examples/complex_ops.cpp`](examples/complex_ops.cpp) | `dznrm2`, `zdscal`, `zaxpy`, `zdotc` |
| [`examples/benchmark.cpp`](examples/benchmark.cpp) | All real and complex routines, memory bandwidth |

Examples are built automatically. To disable them:

```bash
cmake -S . -B build -DTHEBLAS_BUILD_EXAMPLES=OFF
```

## Requirements

| Requirement | Minimum version |
| --- | --- |
| C++ standard | C++17 |
| CMake | 3.15 |
| GCC | 5.0 |
| Clang | 3.5 |
| MSVC | VS 2017 |

Ninja is recommended when using CMake presets. No runtime dependencies beyond the standard library.

### Optional tools

| Tool | Purpose |
| --- | --- |
| `doxygen` + `graphviz` | API documentation with call graphs |
| `cppcheck` | Static analysis |
| `clang-tidy` | Lint and modernization |
| `lcov` | Coverage HTML reports |
| `pre-commit` | Git hook quality gates |
| `arm-linux-gnueabihf-g++` | ARM 32-bit embedded Linux cross-compile |
| `arm-none-eabi-g++` | ARM Cortex-M bare-metal cross-compile |

**Ubuntu/Debian — install everything at once:**

```bash
sudo apt-get install -y cmake ninja-build gcc g++ clang \
  doxygen graphviz cppcheck clang-tidy lcov \
  gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf gcc-arm-none-eabi
```

**macOS (Homebrew):**

```bash
brew install cmake ninja doxygen graphviz cppcheck llvm lcov
```

**Windows:** Install Visual Studio 2017+ with the C++ workload and CMake from <https://cmake.org/download/>.

## Pre-commit

This repository uses [pre-commit](https://pre-commit.com/) hooks to enforce style, formatting, and commit message conventions.

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg
```

Hooks include clang-format, clang-tidy, cmake-format, codespell, markdownlint, and conventional-commit validation. See [.pre-commit-config.yaml](.pre-commit-config.yaml) for the full list.

## Static Analysis

Both `cppcheck` and `clang-tidy` run in CI on every push and pull request.

**cppcheck** (no compilation database required):

```bash
cppcheck --enable=warning,style,performance,portability \
  --std=c++17 --language=c++ --error-exitcode=1 \
  --inline-suppr --suppress=missingIncludeSystem \
  -I include src include tests examples
```

**clang-tidy** (requires a compilation database):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
clang-tidy -p build src/theblas.cpp tests/test_theblas.cpp
```

**scan-build** (browsable HTML report):

```bash
cmake -B build-scan -DCMAKE_BUILD_TYPE=Release
scan-build -o scan-results cmake --build build-scan
```

## Coverage

Generate an lcov HTML coverage report with a single command:

```bash
make coverage
# then open: coverage_html/index.html
```

Or use the CMake preset directly:

```bash
cmake --preset gcc-debug-coverage
cmake --build --preset gcc-debug-coverage
ctest --preset gcc-debug-coverage
```

To enable coverage on any build directory:

```bash
cmake -S . -B build-cov \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTHEBLAS_ENABLE_COVERAGE=ON \
  -DTHEBLAS_ENABLE_IPO=OFF
```

> [!NOTE]
> Disable `THEBLAS_ENABLE_IPO` when using coverage — LTO conflicts with `--coverage`.

## Profiling

`examples/benchmark.cpp` measures each routine and reports achieved memory bandwidth. Build in Release mode:

```bash
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
./build-release/examples/example_benchmark          # default: 1M elements (L3/DRAM)
./build-release/examples/example_benchmark 256      # L1 cache
./build-release/examples/example_benchmark 16384    # L2 cache
```

For instruction-level profiling:

```bash
# perf
perf stat -e cycles,instructions,cache-misses \
  ./build-release/examples/example_benchmark

# Callgrind
valgrind --tool=callgrind ./build-release/examples/example_benchmark 65536
kcachegrind callgrind.out.*
```

## Build

### CMake Presets (recommended)

[CMakePresets.json](CMakePresets.json) provides ready-made configure, build, and test profiles.

```bash
cmake --list-presets
```

Common workflows:

```bash
# GCC debug + sanitizers
cmake --preset gcc-debug-sanitized
cmake --build --preset gcc-debug-sanitized
ctest --preset gcc-debug-sanitized

# Clang release with LTO
cmake --preset clang-release
cmake --build --preset clang-release
ctest --preset clang-release
```

Available presets: `gcc-debug`, `gcc-release`, `gcc-debug-sanitized`, `gcc-debug-coverage`,
`clang-debug`, `clang-debug-sanitized`, `clang-release`, `msvc-release`,
`arm-linux-debug`, `arm-linux-release`, `arm-none-eabi-debug`, `arm-none-eabi-release`.

### Manual build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

### CMake options

| Option | Default | Description |
| --- | --- | --- |
| `THEBLAS_ENABLE_STRICT_WARNINGS` | `ON` | Enable strict compiler warnings |
| `THEBLAS_WARNINGS_AS_ERRORS` | `OFF` | Promote warnings to errors |
| `THEBLAS_ENABLE_SANITIZERS` | `OFF` | ASan + UBSan (Debug, GCC/Clang only) |
| `THEBLAS_ENABLE_COVERAGE` | `OFF` | `--coverage -O0`; disable IPO when on |
| `THEBLAS_ENABLE_IPO` | `ON` | LTO for Release configurations |
| `THEBLAS_ENABLE_RELEASE_HARDENING` | `ON` | `_FORTIFY_SOURCE=3` + stack protector |

### Toolchain files

```bash
# Clang
cmake -S . -B build-clang \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/clang.cmake \
  -DCMAKE_BUILD_TYPE=Release

# MSVC (Visual Studio Developer Command Prompt)
cmake -S . -B build-msvc -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/msvc.cmake
cmake --build build-msvc --config Release

# ARM 32-bit embedded Linux
cmake --preset arm-linux-release

# ARM Cortex-M bare-metal (change MCU as needed)
cmake -S . -B build-arm-bare \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DTHEBLAS_ARM_MCU=cortex-m4
```

### Makefile shortcuts

```bash
make build     # configure + build (default preset: gcc-debug)
make test      # run ctest
make install   # install to ./install
make docs      # generate Doxygen HTML
make coverage  # build + run tests + lcov HTML report
make clean
```

Variables: `BUILD_DIR` (default `build`), `CONFIG` (default `Release`), `PREFIX` (default `./install`).

## Dev Container / Codespaces

This repository ships a [Dev Container](.devcontainer/) configuration with C++/CMake, Doxygen, Graphviz, and all static analysis tools pre-installed.

- **VS Code:** `Dev Containers: Reopen in Container`
- **GitHub Codespaces:** `Code → Codespaces → Create codespace on master`

The post-create command runs `cmake -S . -B build && cmake --build build && ctest --test-dir build` automatically.

## Use with Conan

theblas ships a `conanfile.py` at the repository root, compatible with
[Conan 2.x](https://conan.io).

### Install Conan (one-time)

```bash
pip install conan
conan profile detect
```

### Option A — install directly from source

```bash
git clone https://github.com/thekyria/theblas
cd theblas
conan create . --build=missing
```

This builds the library, runs the `test_package` smoke test, and installs it
into the local Conan cache as `theblas/0.1.0`.

### Option B — consume as a dependency

In your project's `conanfile.py`:

```python
from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout

class MyProject(ConanFile):
    requires = "theblas/0.1.0"
    generators = "CMakeDeps", "CMakeToolchain"
    settings = "os", "arch", "compiler", "build_type"

    def layout(self):
        cmake_layout(self)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
```

Or in a `conanfile.txt`:

```ini
[requires]
theblas/0.1.0

[generators]
CMakeDeps
CMakeToolchain
```

Then in your `CMakeLists.txt`:

```cmake
find_package(theblas CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE theblas::theblas)
```

Install dependencies and build:

```bash
conan install . --build=missing
cmake --preset conan-release   # or the preset generated by CMakeToolchain
cmake --build build/Release
```

## Use with vcpkg

theblas ships its own [vcpkg](https://vcpkg.io) port and versions database, so
it can be consumed directly as a git registry — no submission to the central
vcpkg registry required.

### 1. Install vcpkg (one-time)

```bash
git clone https://github.com/microsoft/vcpkg "$HOME/vcpkg"
"$HOME/vcpkg/bootstrap-vcpkg.sh" -disableMetrics
```

### 2. Add a manifest to your project

**`vcpkg.json`** (alongside your `CMakeLists.txt`):

```json
{
  "name": "my-project",
  "version": "0.1.0",
  "dependencies": ["theblas"]
}
```

**`vcpkg-configuration.json`** (same directory):

```json
{
  "registries": [
    {
      "kind": "git",
      "repository": "https://github.com/thekyria/theblas",
      "baseline": "<commit-hash>",
      "packages": ["theblas"]
    }
  ]
}
```

Replace `<commit-hash>` with the latest commit SHA from the
[theblas repository](https://github.com/thekyria/theblas/commits/master)
that contains the `versions/` directory (any commit after the initial port
was added).

### 3. Consume the target in CMake

```cmake
find_package(theblas CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE theblas::theblas)
```

### 4. Configure and build

```bash
cmake -B build -DCMAKE_TOOLCHAIN_FILE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build
```

vcpkg will automatically download, build, and install theblas before your
project is configured.

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

## API Documentation

Documentation is built and published to [GitHub Pages](https://thekyria.github.io/theblas/) automatically on every push to `master`. To build locally:

```bash
doxygen Doxyfile
# open docs/doxygen/html/index.html
```
