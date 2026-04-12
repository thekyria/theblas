cmake_minimum_required(VERSION 3.15)
project(theblas VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(THEBLAS_BUILD_TESTS "Build theblas tests" ${PROJECT_IS_TOP_LEVEL})

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(theblas
    src/theblas.cpp
)

add_library(theblas::theblas ALIAS theblas)

target_include_directories(theblas
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(TARGETS theblas
    EXPORT theblasTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/theblasConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/theblasConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/theblasConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/theblas
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/theblasConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/theblasConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/theblas
)

install(EXPORT theblasTargets
    FILE theblasTargets.cmake
    NAMESPACE theblas::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/theblas
)

if(THEBLAS_BUILD_TESTS)
    include(CTest)
endif()

if(THEBLAS_BUILD_TESTS AND BUILD_TESTING)
    add_subdirectory(tests)
endif()
