# Compiled as C (not C++) so the public header is checked as real C.
add_executable(test_theblas_c smoke.c)
target_link_libraries(test_theblas_c PRIVATE theblas_c)
set_target_properties(test_theblas_c PROPERTIES C_STANDARD 99
                                                C_STANDARD_REQUIRED ON)
if(MSVC)
  target_compile_options(test_theblas_c PRIVATE /W4 /UNDEBUG)
else()
  target_compile_options(test_theblas_c PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
endif()
add_test(NAME theblas_c_smoke COMMAND test_theblas_c)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME theblas_c_coverage
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_coverage.py
                   ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
  add_test(NAME theblas_c_ctypes
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ctypes_smoke.py
                   $<TARGET_FILE:theblas_c>)
endif()
