add_executable(
	test_theblas
	test_main.cpp
	test_level1.cpp
	test_level2_core.cpp
	test_level2_variants.cpp)

target_link_libraries(test_theblas PRIVATE theblas)

if(MSVC)
	target_compile_options(test_theblas PRIVATE /UNDEBUG)
else()
	target_compile_options(test_theblas PRIVATE -UNDEBUG)
endif()

add_test(NAME theblas_test COMMAND test_theblas)

if(THEBLAS_TEST_AGAINST_NETLIB)
	add_executable(test_theblas_netlib test_against_netlib.cpp)
	target_include_directories(test_theblas_netlib
	                           PRIVATE ${reference_lapack_SOURCE_DIR}/CBLAS/include
	                                   ${reference_lapack_BINARY_DIR}/include)
	target_link_libraries(test_theblas_netlib PRIVATE theblas
	                                                 ${THEBLAS_NETLIB_CBLAS_TARGET})
	if(MSVC)
		target_compile_options(test_theblas_netlib PRIVATE /UNDEBUG)
	else()
		target_compile_options(test_theblas_netlib PRIVATE -UNDEBUG)
	endif()
	add_test(NAME theblas_netlib_test COMMAND test_theblas_netlib)
endif()
