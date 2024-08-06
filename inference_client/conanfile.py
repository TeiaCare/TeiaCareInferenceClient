#!/usr/bin/env python
# Copyright 2024 TeiaCare
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from conans import ConanFile

class GRPC(ConanFile):
    requires = "grpc/1.65.0"
    generators = "CMakeDeps"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            del self.options.fPIC

        self.options["grpc"].codegen=True
        self.options["grpc"].csharp_ext=False
        self.options["grpc"].cpp_plugin=True
        self.options["grpc"].csharp_plugin=False
        self.options["grpc"].node_plugin=False
        self.options["grpc"].objective_c_plugin=False
        self.options["grpc"].php_plugin=False
        self.options["grpc"].python_plugin=False
        self.options["grpc"].ruby_plugin=False
        self.options["grpc"].secure=False

        if self.settings.os == "Linux":
            self.options["grpc"].with_libsystemd=False
