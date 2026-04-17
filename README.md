# theblas

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12538/badge)](https://www.bestpractices.dev/projects/12538)
[![CMake on a single platform](https://github.com/thekyria/theblas/actions/workflows/build.yml/badge.svg)](https://github.com/thekyria/theblas/actions/workflows/build.yml)
[![Coverage](https://github.com/thekyria/theblas/actions/workflows/coverage.yml/badge.svg)](https://github.com/thekyria/theblas/actions/workflows/coverage.yml)
[![Coverage %](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/thekyria/8e1037ada9fe1e598aefebfe56b920ca/raw/theblas-coverage.json)](https://github.com/thekyria/theblas/actions/workflows/coverage.yml)
[![SonarCloud](https://sonarcloud.io/api/project_badges/measure?project=thekyria_theblas&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=thekyria_theblas)
[![CodeQL](https://github.com/thekyria/theblas/actions/workflows/codeql.yml/badge.svg)](https://github.com/thekyria/theblas/actions/workflows/codeql.yml)
[![MSVC](https://github.com/thekyria/theblas/actions/workflows/msvc.yml/badge.svg)](https://github.com/thekyria/theblas/actions/workflows/msvc.yml)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-3.15%2B-blue.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

## Examples

The `examples/` folder contains standalone programs demonstrating library usage.

| Example | Routines covered |
| --- | --- |
| [`examples/vector_ops.cpp`](examples/vector_ops.cpp) | `snrm2`, `sscal`, `saxpy`, `sdot`, `isamax` |
| [`examples/complex_ops.cpp`](examples/complex_ops.cpp) | `dznrm2`, `zdscal`, `zaxpy`, `zdotc` |
| [`examples/benchmark.cpp`](examples/benchmark.cpp) | Microbenchmark: all real + complex routines, bandwidth reporting |

Examples are built automatically when building the project as the top-level CMake project. To disable:

```bash
cmake -S . -B build -DTHEBLAS_BUILD_EXAMPLES=OFF
```

Run them after building:

```bash
./build/gcc-debug/examples/example_vector_ops
./build/gcc-debug/examples/example_complex_ops
```

## Dependencies

### Minimum Requirements

- **C++17** or later
- **CMake 3.15** or later
- A C++ compiler with C++17 support:
  - GCC 5.0+
  - Clang 3.5+
  - MSVC 2017+

### Build Tools

- **Ninja** (recommended for presets workflow; optional if using Make or IDE generators)

### Compiler-Specific

**GCC/Clang:**

- `gcc` / `g++` or `clang` / `clang++`
- Standard C++ runtime library (libstdc++ for GCC, libc++ for Clang)

**MSVC (Windows):**

- Visual Studio 2017 or later with C++ workload
- Build tools can be installed via Visual Studio installer

### Optional Tools

**Doxygen Documentation:**

- `doxygen` (for generating API documentation from source comments)
- `graphviz` (optional, for call graphs in Doxygen output)

**Cross-Compilation to ARM 32-bit (Embedded Linux):**

- `arm-linux-gnueabihf-gcc` / `arm-linux-gnueabihf-g++`
- On Ubuntu/Debian: `sudo apt-get install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf`

**Cross-Compilation to ARM Cortex-M (Bare-Metal):**

- `arm-none-eabi-gcc` / `arm-none-eabi-g++`
- Download from: [ARM GNU Toolchain](https://developer.arm.com/downloads/-/gnu-rm)
- Or on Ubuntu/Debian: `sudo apt-get install gcc-arm-none-eabi`

**Testing & Sanitizers:**

- Built-in support for AddressSanitizer (ASan) and UndefinedBehaviorSanitizer (UBSan) on GCC/Clang
- No additional dependencies required; sanitizers are controlled via CMake options

**Static Analysis:**

- `cppcheck` (for static C++ analysis; warning, style, performance, portability checks)
- Install on Ubuntu/Debian: `sudo apt-get install cppcheck`
- Install on macOS: `brew install cppcheck`
- `clang-tidy` (for AST-based lint, modernization, and bug-prone pattern checks)
- Install on Ubuntu/Debian: `sudo apt-get install clang-tidy`
- Install on macOS: `brew install llvm` (provides `clang-tidy`)
- `scan-build` (Clang Static Analyzer wrapper; produces HTML bug reports)
- Install on Ubuntu/Debian: `sudo apt-get install clang-tools`
- Install on macOS: included with `brew install llvm`

**Pre-commit (Optional, recommended for contributors):**

- `pre-commit` (for automated code quality and Git hook checks)
- Install: `pip install pre-commit`

### Installation Examples

**Ubuntu/Debian (GCC, Clang, Ninja):**

```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build gcc g++ clang
# Optional: for Doxygen and cross-compile
sudo apt-get install -y doxygen graphviz gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf gcc-arm-none-eabi
```

**macOS (GCC/Clang via Homebrew):**

```bash
brew install cmake ninja clang
# Optional:
brew install doxygen graphviz
```

**Windows (MSVC + Visual Studio):**

1. Install Visual Studio 2017 or later with C++ workload.
2. Install CMake from <https://cmake.org/download/> or via `choco install cmake` (Chocolatey).
3. (Optional) Install Ninja: `choco install ninja` or build from source.

## Pre-commit

This repository uses [pre-commit](https://pre-commit.com/) to enforce code quality and Git best practices.

### Setup

Install `pre-commit` and activate Git hooks:

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg
```

### Run Pre-commit Checks

Run all checks on staged files before committing:

```bash
pre-commit run
```

Run all checks on all files:

```bash
pre-commit run --all-files
```

### Included Hooks

- **Trailing whitespace** and **end-of-file fixer**
- **YAML, JSON, TOML validation**
- **Spell checking** with codespell
- **CMake formatting and linting** (cmake-format, cmake-lint)
- **C++ code formatting** with clang-format
- **Conventional commit** validation
- **Markdown linting**

See [.pre-commit-config.yaml](.pre-commit-config.yaml) for full configuration and optional hooks.

## Profiling

### Microbenchmark

`examples/benchmark.cpp` measures each Level-1 routine using `std::chrono`
timing loops (5 warm-up + 20 timed runs, best time reported) and prints
achieved memory bandwidth. Build in Release mode for meaningful numbers:

```bash
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
./build-release/examples/example_benchmark
```

Sweep vector sizes to observe cache hierarchy effects:

```bash
./build-release/examples/example_benchmark 256       # L1 cache
./build-release/examples/example_benchmark 16384     # L2 cache
./build-release/examples/example_benchmark 1048576   # L3 / DRAM (default)
```

### Instruction-Level Profiling with perf

No recompilation required. Records CPU events and call graphs:

```bash
# Hardware event counters
perf stat -e cycles,instructions,cache-misses \
  ./build-release/examples/example_benchmark

# Flame graph / call graph
perf record -g ./build-release/examples/example_benchmark
perf report
```

### Valgrind / Callgrind

Instruction-accurate simulation; slower but no hardware counter required:

```bash
valgrind --tool=callgrind \
  ./build-release/examples/example_benchmark 65536
kcachegrind callgrind.out.*
```

### gprof (GCC instrumentation)

```bash
cmake -B build-gprof -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-pg" -DCMAKE_EXE_LINKER_FLAGS="-pg"
cmake --build build-gprof
./build-gprof/examples/example_benchmark
gprof build-gprof/examples/example_benchmark gmon.out | less
```

## Static Analysis

This repository uses both [cppcheck](https://cppcheck.sourceforge.io/) and
[clang-tidy](https://clang.llvm.org/extra/clang-tidy/) for static analysis.
Both tools run in CI on every push and pull request.

### cppcheck

Covers data-flow bugs, style, performance, and portability. Does not require
a compilation database.

```bash
cppcheck --enable=warning,style,performance,portability \
  --std=c++17 --language=c++ --error-exitcode=1 \
  --inline-suppr --suppress=missingIncludeSystem \
  -I include src include tests examples
```

A non-zero exit code means findings were reported. Use `// cppcheck-suppress <id>`
inline comments to suppress false positives where necessary.

### clang-tidy

Covers AST-based checks: bug-prone patterns, modernization, performance, and
readability. Requires a compilation database.

First configure CMake with compile commands export enabled:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
```

Then run clang-tidy:

```bash
clang-tidy -p build \
  src/theblas.cpp \
  tests/test_theblas.cpp \
  examples/vector_ops.cpp \
  examples/complex_ops.cpp
```

The [`.clang-tidy`](.clang-tidy) file at the repository root controls which
checks are enabled. Use `// NOLINT(<check-name>)` inline comments to suppress
false positives on specific lines.

### scan-build

Wraps the build with the Clang Static Analyzer and produces an HTML report.
Because it intercepts the compiler during the build, it must run against a
fresh (uncached) build directory.

```bash
cmake -B build-scan -DCMAKE_BUILD_TYPE=Release
scan-build -o scan-results cmake --build build-scan
```

Open the generated report:

```bash
xdg-open scan-results/*/index.html   # Linux
open scan-results/*/index.html        # macOS
```

`scan-build` runs the same Clang Static Analyzer passes already covered by
`clang-analyzer-*` in `.clang-tidy`, but produces a browsable HTML report
useful for visual inspection of analysis paths.

## Coverage

Code coverage is measured with **gcov** and **lcov** (HTML report) on GCC/Clang.

### Prerequisites

Ubuntu/Debian:

```bash
sudo apt-get install -y lcov
```

macOS (Homebrew):

```bash
brew install lcov
```

### Generate the HTML report (one command)

```bash
make coverage
```

This will:

1. Configure with `cmake --preset gcc-debug-coverage` (`-O0 --coverage`).
2. Build the library and test binary.
3. Reset any stale counters with `lcov --zerocounters`.
4. Run the test suite via `ctest`.
5. Capture counters with `lcov --capture` (system headers and `tests/` excluded).
6. Render an interactive HTML report into `coverage_html/`.

Open the report:

```bash
xdg-open coverage_html/index.html   # Linux
open coverage_html/index.html        # macOS
```

### Manual step-by-step

```bash
cmake --preset gcc-debug-coverage
cmake --build --preset gcc-debug-coverage
lcov --directory build/gcc-debug-coverage --zerocounters
ctest --preset gcc-debug-coverage
lcov --capture --directory build/gcc-debug-coverage \
     --output-file build/gcc-debug-coverage/coverage.info \
     --exclude '/usr/*' --exclude '*/tests/*'
genhtml build/gcc-debug-coverage/coverage.info \
        --output-directory coverage_html
```

### CMake option

Coverage instrumentation can also be enabled on any build directory:

```bash
cmake -S . -B build-cov \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTHEBLAS_ENABLE_COVERAGE=ON \
  -DTHEBLAS_ENABLE_IPO=OFF
cmake --build build-cov
ctest --test-dir build-cov
```

Note: `THEBLAS_ENABLE_IPO` should be `OFF` when coverage is enabled because
link-time optimisation conflicts with `--coverage`.

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
cmake --preset gcc-release
cmake --build --preset gcc-release
ctest --preset gcc-release
```

```bash
cmake --preset gcc-debug-sanitized
cmake --build --preset gcc-debug-sanitized
ctest --preset gcc-debug-sanitized
```

```bash
cmake --preset clang-debug
cmake --build --preset clang-debug
ctest --preset clang-debug
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
- `gcc-release`
- `gcc-debug-sanitized`
- `gcc-debug-coverage`
- `clang-debug`
- `clang-debug-sanitized`
- `clang-release`
- `msvc-release`
- `arm-linux-debug`
- `arm-linux-release`
- `arm-none-eabi-debug`
- `arm-none-eabi-release`

The `msvc-release` preset is intended for Windows in a Visual Studio Developer Command Prompt.
The `arm-linux-*` presets are for ARM 32-bit (hard-float EABI) embedded Linux targets and require a compatible ARM cross-compiler toolchain.
The `arm-none-eabi-*` presets are for ARM Cortex-M bare-metal embedded systems and require the ARM EABI cross-compiler.

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
- `THEBLAS_ENABLE_COVERAGE=ON|OFF` (default: `OFF`, enables `--coverage -O0 -g`; GCC/Clang only; disable `THEBLAS_ENABLE_IPO` when using this)
- `THEBLAS_ENABLE_IPO=ON|OFF` (default: `ON`, enables interprocedural optimization/LTO for `Release`, `RelWithDebInfo`, and `MinSizeRel` when supported)
- `THEBLAS_ENABLE_RELEASE_HARDENING=ON|OFF` (default: `ON`, enables `_FORTIFY_SOURCE=3` and stack protector for non-Debug Clang/GCC builds)

Release configurations use the compiler toolchain's standard CMake optimization defaults and then add IPO/LTO when available. This avoids overriding compiler-specific defaults such as GCC/Clang `Release` optimization levels.
On Linux with Clang, IPO/LTO additionally requires a compatible LLVM linker such as `ld.lld`; if it is not available, the project falls back to the normal compiler release optimization defaults.

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

Release example:

```bash
cmake -S . -B build-release \
 -DCMAKE_BUILD_TYPE=Release \
 -DTHEBLAS_ENABLE_IPO=ON \
 -DTHEBLAS_ENABLE_RELEASE_HARDENING=ON
cmake --build build-release
ctest --test-dir build-release
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

### Cross-compile for ARM 32-bit (Embedded Linux)

This repository includes a toolchain file for ARM 32-bit hard-float EABI embedded Linux: `cmake/toolchains/arm-linux-gnueabihf.cmake`.

Prerequisites:

- ARM cross-compiler toolchain (e.g., `arm-linux-gnueabihf-g++`, `arm-linux-gnueabihf-gcc`)
- On Ubuntu/Debian: `sudo apt-get install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf`

Using presets:

```bash
cmake --preset arm-linux-debug
cmake --build --preset arm-linux-debug
```

```bash
cmake --preset arm-linux-release
cmake --build --preset arm-linux-release
```

Manual cross-compile:

```bash
cmake -S . -B build-arm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-linux-gnueabihf.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build-arm
```

To use a custom ARM sysroot, edit `cmake/toolchains/arm-linux-gnueabihf.cmake` and uncomment the `THEBLAS_ARM_SYSROOT` section, or pass it on the command line:

```bash
cmake -S . -B build-arm \
 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-linux-gnueabihf.cmake \
 -DCMAKE_BUILD_TYPE=Release
cmake --build build-arm
```

### Cross-compile for ARM Cortex-M (Bare-Metal)

This repository includes a toolchain file for ARM Cortex-M bare-metal development: `cmake/toolchains/arm-none-eabi.cmake`.

Prerequisites:

- ARM EABI cross-compiler toolchain (e.g., `arm-none-eabi-gcc`, `arm-none-eabi-g++`)
- Download from: [ARM Developer GCC](https://developer.arm.com/downloads/-/gnu-rm)
- Or on Ubuntu/Debian: `sudo apt-get install gcc-arm-embedded`

Using presets:

```bash
cmake --preset arm-none-eabi-debug
cmake --build --preset arm-none-eabi-debug
```

```bash
cmake --preset arm-none-eabi-release
cmake --build --preset arm-none-eabi-release
```

Manual cross-compile with MCU selection (default is cortex-m4):

```bash
cmake -S . -B build-arm-bare \
 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi.cmake \
 -DCMAKE_BUILD_TYPE=Release \
 -DTHEBLAS_ARM_MCU=cortex-m4
cmake --build build-arm-bare
```

To target a different Cortex-M variant, adjust `-DTHEBLAS_ARM_MCU` (e.g., `cortex-m0`, `cortex-m3`, `cortex-m7`, `cortex-m33`).

## Build with Makefile

```bash
make help
make build
make test
make install
make docs
make clean
make coverage
```

Useful variables:

- `BUILD_DIR` (default: `build`)
- `CONFIG` (default: `Release`)
- `PREFIX` (default: `./install`)
- `COVERAGE_DIR` (default: `build/gcc-debug-coverage`)
- `COVERAGE_HTML` (default: `coverage_html`)

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
