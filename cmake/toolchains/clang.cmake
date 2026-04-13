# Clang toolchain file for local/native builds.
# Usage: cmake -S . -B build-clang -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/clang.cmake

set(CMAKE_C_COMPILER clang CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER clang++ CACHE STRING "C++ compiler" FORCE)
