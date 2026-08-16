add_executable(
	test_theblas
	test_main.cpp
	test_level1.cpp
	test_level2_core.cpp
	test_level2_variants.cpp)

target_link_libraries(test_theblas PRIVATE theblas)

add_test(NAME theblas_test COMMAND test_theblas)
