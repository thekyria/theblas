# Changelog

All notable changes to theblas are documented here.

This project follows [Semantic Versioning 2.0.0](https://semver.org/):
- **MAJOR** — incompatible API changes
- **MINOR** — new functionality, backward-compatible
- **PATCH** — backward-compatible bug fixes

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

## [0.1.0] - 2026-04-13

### Added

- Initial Level-1 BLAS implementation in C++17 (`theblas` namespace):
  - Real single/double: `sswap`, `dswap`, `scopy`, `dcopy`, `saxpy`, `daxpy`,
    `sscal`, `dscal`, `sdot`, `ddot`, `snrm2`, `dnrm2`, `sasum`, `dasum`,
    `isamax`, `idamax`
  - Complex single/double: `cswap`, `zswap`, `ccopy`, `zcopy`, `caxpy`, `zaxpy`,
    `cscal`, `zscal`, `csscal`, `zdscal`, `cdotu`, `cdotc`, `zdotu`, `zdotc`,
    `scnrm2`, `dznrm2`, `scasum`, `dzasum`, `icamax`, `izamax`
  - Plane rotations: `srot`, `drot`, `csrot`, `zdrot`
  - Givens generation: `srotg`, `drotg`, `crotg`, `zrotg`
  - Modified Givens: `srotm`, `drotm`, `srotmg`, `drotmg`
- Compile-time version macros via generated `theblas/version.h`:
  - `THEBLAS_VERSION_MAJOR`, `THEBLAS_VERSION_MINOR`, `THEBLAS_VERSION_PATCH`
  - `THEBLAS_VERSION_STRING`, `THEBLAS_VERSION` (integer encoding)
- CMake build system with presets (GCC, Clang, MSVC, ARM cross-compile)
- CMake package config (`find_package(theblas CONFIG REQUIRED)`)
- CTest unit test suite
- Examples: `vector_ops`, `complex_ops`, `benchmark`
- Static analysis: cppcheck, clang-tidy, scan-build
- Pre-commit hooks (clang-format, cmake-format, codespell, conventional commit)
- CI via GitHub Actions

[Unreleased]: https://github.com/thekyria_github/theblas/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/thekyria_github/theblas/releases/tag/v0.1.0
