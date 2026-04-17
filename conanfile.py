from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy, rmdir
import os


class TheBlasConan(ConanFile):
    name = "theblas"
    version = "0.1.0"
    description = "A minimal C++17 Level-1 BLAS library"
    license = "MIT"
    url = "https://github.com/thekyria/theblas"
    homepage = "https://github.com/thekyria/theblas"
    topics = ("blas", "linear-algebra", "math", "cpp17")

    # Sources are kept in the same place as the recipe
    exports_sources = (
        "CMakeLists.txt",
        "src/*",
        "include/*",
        "cmake/*",
    )

    settings = "os", "arch", "compiler", "build_type"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["THEBLAS_BUILD_TESTS"] = False
        tc.variables["THEBLAS_ENABLE_STRICT_WARNINGS"] = False
        tc.variables["THEBLAS_WARNINGS_AS_ERRORS"] = False
        tc.variables["THEBLAS_ENABLE_SANITIZERS"] = False
        tc.variables["THEBLAS_ENABLE_COVERAGE"] = False
        tc.variables["THEBLAS_ENABLE_IPO"] = False
        tc.variables["THEBLAS_ENABLE_RELEASE_HARDENING"] = False
        tc.variables["BUILD_SHARED_LIBS"] = self.options.shared
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        copy(self, "LICENSE", self.source_folder, os.path.join(self.package_folder, "licenses"))
        cmake = CMake(self)
        cmake.install()
        # Remove CMake package config files — Conan manages discovery itself
        rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "theblas")
        self.cpp_info.set_property("cmake_target_name", "theblas::theblas")
        self.cpp_info.libs = ["theblas"]
