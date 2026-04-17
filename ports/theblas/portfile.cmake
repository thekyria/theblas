vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  thekyria/theblas
  REF
  "v${VERSION}"
  SHA512
  c63dd2d1262f8bbc35640c5caeb57d69c9053eb65a6855142b9308c2a73c688f90f9c97d928f705f7c831ebea6049c57b0d749bec306f9e5ce3191e42facea08
  HEAD_REF
  master)

vcpkg_cmake_configure(
  SOURCE_PATH
  "${SOURCE_PATH}"
  OPTIONS
  -DTHEBLAS_BUILD_TESTS=OFF
  -DTHEBLAS_ENABLE_STRICT_WARNINGS=OFF
  -DTHEBLAS_WARNINGS_AS_ERRORS=OFF
  -DTHEBLAS_ENABLE_SANITIZERS=OFF
  -DTHEBLAS_ENABLE_COVERAGE=OFF
  -DTHEBLAS_ENABLE_IPO=OFF
  -DTHEBLAS_ENABLE_RELEASE_HARDENING=OFF)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME theblas CONFIG_PATH lib/cmake/theblas)

# Remove the debug include directory (headers are the same for all configs)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

# Install license
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")

# Provide a usage message shown after installation
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
