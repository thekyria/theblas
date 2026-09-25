#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "theblas::theblas" for configuration "Release"
set_property(TARGET theblas::theblas APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(theblas::theblas PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libtheblas.a"
  )

list(APPEND _cmake_import_check_targets theblas::theblas )
list(APPEND _cmake_import_check_files_for_theblas::theblas "${_IMPORT_PREFIX}/lib/libtheblas.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
