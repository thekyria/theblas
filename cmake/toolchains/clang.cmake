# Clang toolchain file for local/native builds. Usage: cmake -S . -B build-clang
# -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/clang.cmake

set(CMAKE_C_COMPILER
    clang
    CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER
    clang++
    CACHE STRING "C++ compiler" FORCE)

find_program(THEBLAS_LLVM_AR NAMES llvm-ar)
if(THEBLAS_LLVM_AR)
  set(CMAKE_AR
      "${THEBLAS_LLVM_AR}"
      CACHE FILEPATH "Archiver" FORCE)
  set(CMAKE_C_COMPILER_AR
      "${THEBLAS_LLVM_AR}"
      CACHE FILEPATH "C compiler archiver" FORCE)
  set(CMAKE_CXX_COMPILER_AR
      "${THEBLAS_LLVM_AR}"
      CACHE FILEPATH "CXX compiler archiver" FORCE)
endif()

find_program(THEBLAS_LLVM_RANLIB NAMES llvm-ranlib)
if(THEBLAS_LLVM_RANLIB)
  set(CMAKE_RANLIB
      "${THEBLAS_LLVM_RANLIB}"
      CACHE FILEPATH "Ranlib" FORCE)
  set(CMAKE_C_COMPILER_RANLIB
      "${THEBLAS_LLVM_RANLIB}"
      CACHE FILEPATH "C compiler ranlib" FORCE)
  set(CMAKE_CXX_COMPILER_RANLIB
      "${THEBLAS_LLVM_RANLIB}"
      CACHE FILEPATH "CXX compiler ranlib" FORCE)
endif()

find_program(THEBLAS_LLVM_NM NAMES llvm-nm)
if(THEBLAS_LLVM_NM)
  set(CMAKE_NM
      "${THEBLAS_LLVM_NM}"
      CACHE FILEPATH "Symbol dumper" FORCE)
endif()
