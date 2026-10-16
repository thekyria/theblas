vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  thekyria/theblas
  REF
  "v${VERSION}"
  SHA512
  48b3b3b76a5605b3e137c1829e7575a6bfdac995fb2967e6ca72ca2ddc0e17befede7cc5b9a29852fe43619fa40a7c82478ae0cf56efbe545ecefc0aedb2df15
  HEAD_REF
  master)

vcpkg_cmake_configure(
  SOURCE_PATH
  "${SOURCE_PATH}"
  OPTIONS
  -DTHEBLAS_BUILD_TESTS=OFF
  -DTHEBLAS_BUILD_EXAMPLES=OFF
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
