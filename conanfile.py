#!/usr/bin/env python
from conans import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain

def get_project_version():
    with open('VERSION') as version_file:
        # TODO: validate Regex format
        return version_file.read().strip()

class TeiaCareInferenceClient(ConanFile):
    name = "teiacare_inference_client"
    version = get_project_version()
    author = "TeiaCare"
    url = "https://github.com/TeiaCare/TeiaCareInferenceClient"
    description = "TeiaCareInferenceClient is a C++ inference client library that implements KServe protocol"
    topics = ("inference_client", "kserve")
    exports = "VERSION"
    exports_sources = "CMakeLists.txt", "inference_client/CMakeLists.txt", "inference_client/include/*", "inference_client/src/*", "cmake/*", "proto/services.proto"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}
    requires = "grpc/1.54.3"
    generators = "CMakeDeps"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            del self.options.fPIC

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["BUILD_SHARED_LIBS"] = self.options.shared
        tc.variables["TC_ENABLE_UNIT_TESTS"] = False
        tc.variables["TC_ENABLE_UNIT_TESTS_COVERAGE"] = False
        tc.variables["TC_ENABLE_BENCHMARKS"] = False
        tc.variables["TC_ENABLE_EXAMPLES"] = False
        tc.variables["TC_ENABLE_DOCS"] = False
        tc.variables["TC_ENABLE_WARNINGS_ERROR"] = True
        tc.variables["TC_ENABLE_SANITIZER_ADDRESS"] = False
        tc.variables["TC_ENABLE_SANITIZER_THREAD"] = False
        tc.variables["TC_ENABLE_CLANG_FORMAT"] = False
        tc.variables["TC_ENABLE_CLANG_TIDY"] = False
        tc.variables["TC_ENABLE_CPPCHECK"] = False
        tc.variables["TC_ENABLE_CPPLINT"] = False
        tc.variables["TC_ENABLE_DOCS"] = False
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        self.copy(pattern="VERSION")
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["teiacare_inference_client"]
        self.cpp_info.set_property("cmake_file_name", "teiacare_inference_client")
        self.cpp_info.set_property("cmake_target_name", "teiacare::inference_client")
