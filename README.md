# TeiaCareInferenceClient
Welcome to TeiaCareInferenceClient!

*TeiaCareInferenceClient* is a C++ inference client library that implements KServe protocol.

![TeiaCareInferenceClient](https://socialify.git.ci/TeiaCare/TeiaCareinferenceclient/image?description=1&font=Raleway&name=1&pattern=Solid&theme=Auto)

## Getting Started

### Create a virtual environment

```bash
python -m pip install --upgrade pip
python -m venv .venv

# Linux/MacOS
echo "export CONAN_USER_HOME=$PWD" >> .venv/bin/activate
source .venv/bin/activate

# Windows
echo set CONAN_USER_HOME=%CD%>>.venv\Scripts\activate.bat
.venv\Scripts\activate.bat

pip install -r scripts/requirements.txt
pre-commit install
```

### Setup Build Environment (Windows Only)

When building from command line on Windows it is necessary to activate the Visual Studio Developer Command Prompt.
Depending on the version of Visual Studio compiler and on its install location it is required to run *vcvars64.bat* script the set the development environment properly.
*Note*: using Visual Studio IDE or the CMake extension for VSCode this step is already managed in the background, so no action is required.

Examples:

```bash
# Visual Studio 2022 - Build Tools
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

# Visual Studio 2019 - Enterprise
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
```

### Dependencies Setup

This script must be executed in order to setup the conan packages (note that 3rd party libs are only required for unit tests, examples and benchmarks).
```bash
python scripts/conan/setup.py <Debug|Release|DebWithRelInfo> <COMPILER_NAME> <COMPILER_VERSION>
```

### Configure, Build and Install

This script configures, builds and installs the library.
```bash
python scripts/cmake.py <Debug|Release|DebWithRelInfo> <COMPILER_NAME> <COMPILER_VERSION>
```

## Examples

```bash
# Build all the examples
python scripts/cmake.py <Debug|Release|RelWithDebInfo> <COMPILER_NAME> <COMPILER_VERSION> --examples --warnings

# Run all the examples
python scripts/tools/run_examples.py install/examples
```
Examples are installed in $PWD/install/examples.


## Unit Tests and Code Coverage

```bash
# Build Unit Tests with Code Coverage enabled (if supported)
python scripts/cmake.py <Debug|Release|RelWithDebInfo> <COMPILER_NAME> <COMPILER_VERSION> --coverage --warnings

# Run Unit Tests
python scripts/tools/run_unit_tests.py <Debug|Release|RelWithDebInfo>

# Run Code Covergae
python scripts/tools/run_coverage.py <COMPILER_NAME> <COMPILER_VERSION>
```
Note that code coverage is not available on Windows.

Unit tests results are available in $PWD/results/unit_tests.
Coverage results are available in $PWD/results/coverage.


## Sanitizers

### Address Sanitizer

```bash
# Build Unit Tests with Address Sanitizer enabled (if supported)
python scripts/cmake.py <Debug|Release|RelWithDebInfo> <COMPILER_NAME> <COMPILER_VERSION> --address_sanitizer --unit_tests

# Run Unit Tests with Address Sanitizer
python scripts/tools/run_sanitizer.py --address_sanitizer install/unit_tests/teiacare_inference_client_unit_tests
```
Note that Address Sanitizer is supported only on Linux.


### Thread Sanitizer

```bash
# Build Unit Tests with Thread Sanitizer enabled (if supported)
python scripts/cmake.py <Debug|Release|RelWithDebInfo> <COMPILER_NAME> <COMPILER_VERSION> --thread_sanitizer --unit_tests

# Run Unit Tests with Thread Sanitizer
python scripts/tools/run_sanitizer.py --thread_sanitizer install/unit_tests/teiacare_inference_client_unit_tests
```
Note that Thread Sanitizer is supported only on Linux.


## Benchmarks

```bash
# Build Benkmarks
python scripts/cmake.py <Debug|Release|RelWithDebInfo> <COMPILER_NAME> <COMPILER_VERSION> --benchmarks --warnings

# Run Benchmarks
python scripts/tools/run_benchmarks.py <COMPILER_NAME> <COMPILER_VERSION>
```
Benchmarks are installed in $PWD/install/benchmarks.


## Code Formatting

- [clang-format](https://clang.llvm.org/docs/ClangFormat.html)

*clang-format* can be installed via *pip* using the provided *scripts/requirements.txt*

```bash
python scripts/tools/run_clang_format.py -r -i inference_client
```


## Code Analysis

- [clang-tidy](https://clang.llvm.org/extra/clang-tidy/)

*clang-tidy* can be installed via *pip* using the provided *scripts/requirements.txt*

```bash
python scripts/tools/run_clang_tidy.py -header-filter=.* inference_client
```

- [cppcheck](http://cppcheck.net/)

First install and setup *cppcheck* from your OS package manager.
```bash
# Linux
sudo apt install cppcheck

# Windows
winget install cppcheck
```

Then run CppCheck using the provided python script:
```bash
python scripts/tools/run_cppcheck.py <Debug|Release|RelWithDebInfo>
```


## Generate Documentation

First install and setup *Doxygen* from your OS package manager.
```bash
# Linux
apt-get install doxygen graphviz

# Windows
winget install doxygen
```

Then run Doxygen using the provided python script:
```bash
python scripts/tools/run_doxygen.py
```
Documentation is now installed in $PWD/docs.


## Conan Package

### Local Install

Create, test and install local package.

Notes:
1) The install directory path must be a valid Conan cache (i.e. ".conan" folder) located in the current directory.
   So, in order to install the package in a desired repository folder, it is required to run this script from the repository folder directly.
2) The Conan package tests are automatically run during package creation.
   The directory test_package contains a test project that is built to validate the proper package creation.

```bash
# Create the Conan package locally
python scripts/conan/create.py <Debug|Release|RelWithDebInfo> <COMPILER_NAME> <COMPILER_VERSION>

# Build and install the test package executable
python test_package/build.py <Debug|Release|RelWithDebInfo> <COMPILER_NAME> <COMPILER_VERSION>

# Run the test package executable
$PWD/install/test_package/teiacare_inference_client_test_package
```


### Artifactory Upload

In order to upload a Conan package to TeiaCare Artifactory server it is required to setup you local Conan client once with the following commands:

```bash
# Add TeiaCare Artifactory remote to local Conan client
conan remote add teiacare $(artifactory.url)/teiacare

# Authenticate with Artifactory credentials
conan user $(artifactory.username) -p $(artifactory.password) -r teiacare
```

Now it is possible to create and upload a Conan package with the following commands:

```bash
# Create the Conan package locally
python scripts/conan/create.py <Debug|Release|RelWithDebInfo> <COMPILER_NAME> <COMPILER_VERSION>

# Upload the package to Artifactory on the teicare remote
python scripts/conan/upload.py teiacare teiacare_inference_client
```


## Contributing

In order to contribute to TeiaCareInferenceClient, please follow our [contribution guidelines](./CONTRIBUTING).

[![Contributions](https://img.shields.io/badge/Contributions-Welcome-green.svg)](./CONTRIBUTING)


## License

This project is licensed under the [Apache License, Version 2.0](./LICENSE).
Copyright © 2024 [TeiaCare](https://teiacare.com/)

[![License](https://img.shields.io/badge/License-Apache_v2-blue)](./LICENSE)
