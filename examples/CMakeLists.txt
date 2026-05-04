option(THEBLAS_BUILD_EXAMPLES "Build theblas usage examples"
       ${PROJECT_IS_TOP_LEVEL})

if(THEBLAS_BUILD_EXAMPLES)
  add_executable(example_vector_ops vector_ops.cpp)
  target_link_libraries(example_vector_ops PRIVATE theblas::theblas)

  add_executable(example_complex_ops complex_ops.cpp)
  target_link_libraries(example_complex_ops PRIVATE theblas::theblas)

  add_executable(example_matrix_vector_ops matrix_vector_ops.cpp)
  target_link_libraries(example_matrix_vector_ops PRIVATE theblas::theblas)

  add_executable(example_benchmark benchmark.cpp)
  target_link_libraries(example_benchmark PRIVATE theblas::theblas)
  # Release optimisations are essential for meaningful benchmark numbers.
  target_compile_options(
    example_benchmark PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native>)
endif()
