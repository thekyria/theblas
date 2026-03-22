add_executable(test_theblas
    test_theblas.cpp
)

target_link_libraries(test_theblas PRIVATE theblas)

add_test(NAME theblas_test COMMAND test_theblas)
