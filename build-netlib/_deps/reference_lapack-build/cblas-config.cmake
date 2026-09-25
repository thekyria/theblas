# Load the LAPACK package with which we were built.
set(LAPACK_DIR "/home/runner/work/theblas/theblas/build-netlib/_deps/reference_lapack-build")
find_package(LAPACK NO_MODULE)

# Load lapack targets from the build tree, including lapacke targets.
if(NOT TARGET lapacke)
  include("/home/runner/work/theblas/theblas/build-netlib/_deps/reference_lapack-build/lapack-targets.cmake")
endif()

# Report cblas header search locations from build tree.
set(CBLAS_INCLUDE_DIRS "/home/runner/work/theblas/theblas/build-netlib/_deps/reference_lapack-build/include")

# Report cblas libraries.
set(CBLAS_LIBRARIES cblas)
