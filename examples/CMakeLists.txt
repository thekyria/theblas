option(THEBLAS_BUILD_EXAMPLES "Build theblas usage examples"
       ${PROJECT_IS_TOP_LEVEL})

if(THEBLAS_BUILD_EXAMPLES)
  add_executable(example_vector_ops vector_ops.cpp)
  target_link_libraries(example_vector_ops PRIVATE theblas::theblas)

  add_executable(example_complex_ops complex_ops.cpp)
  target_link_libraries(example_complex_ops PRIVATE theblas::theblas)
endif()
