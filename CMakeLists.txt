cmake_minimum_required(VERSION 3.15)
project(theblas VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(theblas
    src/theblas.cpp
)

target_include_directories(theblas
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
