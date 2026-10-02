file(REMOVE_RECURSE
  "../../lib/libcblas.a"
  "../../lib/libcblas.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C Fortran)
  include(CMakeFiles/cblas.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
