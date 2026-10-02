// NOLINTNEXTLINE(portability-avoid-pragma-once)
#pragma once

/// @defgroup version Version macros
/// @brief Compile-time semantic version constants for theblas.
///
/// The version follows [Semantic Versioning 2.0.0](https://semver.org/):
/// - MAJOR is incremented on incompatible API changes.
/// - MINOR is incremented when functionality is added in a backward-compatible manner.
/// - PATCH is incremented for backward-compatible bug fixes.
/// @{

/// Major version component.
// NOLINTNEXTLINE(modernize-macro-to-enum)
#define THEBLAS_VERSION_MAJOR 0

/// Minor version component.
// NOLINTNEXTLINE(modernize-macro-to-enum)
#define THEBLAS_VERSION_MINOR 2

/// Patch version component.
// NOLINTNEXTLINE(modernize-macro-to-enum)
#define THEBLAS_VERSION_PATCH 0

/// Full version as a null-terminated string literal, e.g. "0.1.0".
#define THEBLAS_VERSION_STRING "0.2.0"

/// Integer encoding: MAJOR * 10000 + MINOR * 100 + PATCH.
/// Suitable for compile-time range checks:
/// @code
///   #if THEBLAS_VERSION >= 10000   // 1.0.0 or later
///   #if THEBLAS_VERSION >= 10100   // 1.1.0 or later
/// @endcode
// NOLINTNEXTLINE(modernize-macro-to-enum)
#define THEBLAS_VERSION \
    (THEBLAS_VERSION_MAJOR * 10000 + THEBLAS_VERSION_MINOR * 100 + THEBLAS_VERSION_PATCH)

/// @}
