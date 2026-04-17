cmake_minimum_required(VERSION 3.15)
project(test_theblas_conan CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(theblas CONFIG REQUIRED)

add_executable(test_theblas_conan main.cpp)
target_link_libraries(test_theblas_conan PRIVATE theblas::theblas)
