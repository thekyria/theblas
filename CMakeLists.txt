cmake_minimum_required(VERSION 3.15)
project(
  theblas
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(THEBLAS_BUILD_TESTS "Build theblas tests" ${PROJECT_IS_TOP_LEVEL})
option(THEBLAS_ENABLE_STRICT_WARNINGS "Enable strict compiler warnings" ON)
option(THEBLAS_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(THEBLAS_ENABLE_SANITIZERS
       "Enable ASan/UBSan for Debug builds (Clang/GCC)" OFF)
option(THEBLAS_ENABLE_IPO "Enable IPO/LTO for release builds when supported" ON)
option(THEBLAS_ENABLE_RELEASE_HARDENING
       "Enable additional hardening for release configurations on Clang/GCC" ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CheckIPOSupported)

add_library(theblas src/theblas.cpp)

add_library(theblas::theblas ALIAS theblas)

target_include_directories(
  theblas PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                 $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(THEBLAS_ENABLE_STRICT_WARNINGS)
  if(MSVC)
    target_compile_options(
      theblas
      PRIVATE /W4
              /permissive-
              /w14242
              /w14254
              /w14263
              /w14265
              /w14287
              /we4289
              /w14296
              /w14311
              /w14545
              /w14546
              /w14547
              /w14549
              /w14555
              /w14619
              /w14640
              /w14826
              /w14905
              /w14906
              /w14928)
    if(THEBLAS_WARNINGS_AS_ERRORS)
      target_compile_options(theblas PRIVATE /WX)
    endif()
  else()
    target_compile_options(
      theblas
      PRIVATE -Wall
              -Wextra
              -Wpedantic
              -Wconversion
              -Wsign-conversion
              -Wshadow
              -Wnon-virtual-dtor
              -Wold-style-cast
              -Woverloaded-virtual
              -Wnull-dereference
              -Wdouble-promotion
              -Wformat=2)
    if(THEBLAS_WARNINGS_AS_ERRORS)
      target_compile_options(theblas PRIVATE -Werror)
    endif()
  endif()
endif()

if(THEBLAS_ENABLE_SANITIZERS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-fno-omit-frame-pointer
                        $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    add_link_options(-fno-omit-frame-pointer
                     $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
  else()
    message(
      WARNING "THEBLAS_ENABLE_SANITIZERS is ON, but sanitizers are configured"
              " only for Clang/GNU.")
  endif()
endif()

if(THEBLAS_ENABLE_IPO)
  set(THEBLAS_ENABLE_IPO_FOR_TARGET TRUE)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT MSVC)
    find_program(THEBLAS_LLD NAMES ld.lld lld)
    if(NOT THEBLAS_LLD)
      set(THEBLAS_ENABLE_IPO_FOR_TARGET FALSE)
      message(STATUS "IPO/LTO disabled for Clang: no compatible LLVM linker"
                     " (ld.lld) was found.")
    endif()
  endif()

  if(THEBLAS_ENABLE_IPO_FOR_TARGET)
    check_ipo_supported(RESULT THEBLAS_IPO_SUPPORTED OUTPUT THEBLAS_IPO_ERROR)
  else()
    set(THEBLAS_IPO_SUPPORTED FALSE)
  endif()

  if(THEBLAS_IPO_SUPPORTED)
    set_property(TARGET theblas PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE
                                         ON)
    set_property(TARGET theblas
                 PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
    set_property(TARGET theblas PROPERTY INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL
                                         ON)
  elseif(THEBLAS_IPO_ERROR)
    message(STATUS "IPO/LTO not enabled: ${THEBLAS_IPO_ERROR}")
  endif()
endif()

if(THEBLAS_ENABLE_RELEASE_HARDENING AND CMAKE_CXX_COMPILER_ID MATCHES
                                        "Clang|GNU")
  target_compile_definitions(
    theblas
    PRIVATE $<$<CONFIG:Release>:_FORTIFY_SOURCE=3>
            $<$<CONFIG:RelWithDebInfo>:_FORTIFY_SOURCE=3>
            $<$<CONFIG:MinSizeRel>:_FORTIFY_SOURCE=3>)
  target_compile_options(
    theblas
    PRIVATE $<$<CONFIG:Release>:-fstack-protector-strong>
            $<$<CONFIG:RelWithDebInfo>:-fstack-protector-strong>
            $<$<CONFIG:MinSizeRel>:-fstack-protector-strong>)
endif()

install(
  TARGETS theblas
  EXPORT theblasTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/theblasConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/theblasConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/theblasConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/theblas)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/theblasConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/theblasConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/theblas)

install(
  EXPORT theblasTargets
  FILE theblasTargets.cmake
  NAMESPACE theblas::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/theblas)

if(THEBLAS_BUILD_TESTS)
  include(CTest)
endif()

if(THEBLAS_BUILD_TESTS AND BUILD_TESTING)
  add_subdirectory(tests)
endif()

add_subdirectory(examples)
