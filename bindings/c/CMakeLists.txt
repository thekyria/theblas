cmake_minimum_required(VERSION 3.15)
project(
  theblas_c
  VERSION 0.3.0
  LANGUAGES C CXX)

# Standalone C ABI shim over theblas. It is not part of the root build.
#
# By default theblas is built from this source tree (as position-independent
# code, so it can be linked into the shared library). Set
# THEBLAS_C_FROM_SOURCE=OFF to use an installed theblas package instead; it must
# then have been built with CMAKE_POSITION_INDEPENDENT_CODE=ON or as a shared
# library.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(THEBLAS_C_FROM_SOURCE "Build theblas from the enclosing source tree" ON)
option(THEBLAS_C_BUILD_TESTS "Build theblas_c tests" ${PROJECT_IS_TOP_LEVEL})

if(THEBLAS_C_FROM_SOURCE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. theblas EXCLUDE_FROM_ALL)
else()
  find_package(theblas CONFIG REQUIRED)
endif()

include(GNUInstallDirs)

add_library(theblas_c SHARED src/theblas_c.cpp)
add_library(theblas::theblas_c ALIAS theblas_c)

target_include_directories(
  theblas_c PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(theblas_c PRIVATE theblas::theblas)
target_compile_definitions(theblas_c PRIVATE THEBLAS_C_EXPORTS)
set_target_properties(theblas_c PROPERTIES CXX_VISIBILITY_PRESET hidden
                                           VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Export only the theblas_* C symbols, not the C++ ones from the static archive.
  target_link_options(theblas_c PRIVATE -Wl,--exclude-libs,ALL)
endif()

if(MSVC)
  target_compile_options(theblas_c PRIVATE /W4 /permissive-)
else()
  target_compile_options(theblas_c PRIVATE -Wall -Wextra -Wpedantic -Wconversion
                                           -Wold-style-cast)
endif()

install(
  TARGETS theblas_c
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/theblas/theblas_c.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/theblas)

if(THEBLAS_C_BUILD_TESTS)
  include(CTest)
  if(BUILD_TESTING)
    add_subdirectory(tests)
  endif()
endif()
