#!/usr/bin/python
import argparse
import subprocess
import os
import sys
from command import run, check_venv

def parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("build_type",           choices=['Debug', 'Release'])
    parser.add_argument("compiler",             help="Compiler name", choices=['gcc', 'clang', 'visual_studio'])
    parser.add_argument("compiler_version",     help="Compiler version")
    parser.add_argument("--build_dir",          required=False, default='./build')
    parser.add_argument("--integration_tests",  required=False, default=False, action='store_true')
    parser.add_argument("--unit_tests",         required=False, default=False, action='store_true')
    parser.add_argument("--coverage",           required=False, default=False, action='store_true')
    parser.add_argument("--benchmarks",         required=False, default=False, action='store_true')
    parser.add_argument("--examples",           required=False, default=False, action='store_true')
    parser.add_argument("--warnings",           required=False, default=False, action='store_true')
    parser.add_argument("--address_sanitizer",  required=False, default=False, action='store_true')
    parser.add_argument("--thread_sanitizer",   required=False, default=False, action='store_true')
    parser.add_argument("--clang_format",       required=False, default=False, action='store_true')
    parser.add_argument("--clang_tidy",         required=False, default=False, action='store_true')
    parser.add_argument("--cppcheck",           required=False, default=False, action='store_true')
    parser.add_argument("--cpplint",            required=False, default=False, action='store_true')
    parser.add_argument("--docs",               required=False, default=False, action='store_true')
    return parser.parse_args()

def main():
    check_venv()
    args = parse()
    profile_name = f'{args.compiler+args.compiler_version}'

    CC = subprocess.run(f'conan profile get env.CC {profile_name}', shell=True, capture_output=True).stdout.decode().strip()
    CXX = subprocess.run(f'conan profile get env.CXX {profile_name}', shell=True, capture_output=True).stdout.decode().strip()

    if not CC or not CXX:
        raise SystemError("\n========================================================"
                          "\nUnable to find CC or CXX environment variable"
                          f"\nPlease check the conan profile {profile_name}"
                          f"\nat {os.getenv('CONAN_USER_HOME')}/.conan/profiles"
                          "\n========================================================\n")
    
    print("\n========================================================")
    print("CONAN_USER_HOME:", os.getenv('CONAN_USER_HOME'))
    print("CXX:", CXX)
    print("CC:", CC)
    print("========================================================\n")

    run([
        'cmake',
        '-G', 'Ninja',
        '-D', f'CMAKE_CC_COMPILER={CC}',
        '-D', f'CMAKE_CXX_COMPILER={CXX}',
        '-D', f'CMAKE_BUILD_TYPE={str(args.build_type)}',
        '-D', f'TC_INFERENCE_CLIENT_ENABLE_WARNINGS_ERROR={str(args.warnings)}',
        '-D', f'TC_INFERENCE_CLIENT_ENABLE_UNIT_TESTS={str(args.unit_tests or args.coverage)}',
        '-D', f'TC_INFERENCE_CLIENT_ENABLE_UNIT_TESTS_COVERAGE={str(args.coverage)}',
        '-D', f'TC_INFERENCE_CLIENT_ENABLE_BENCHMARKS={str(args.benchmarks)}',
        '-D', f'TC_INFERENCE_CLIENT_ENABLE_EXAMPLES={str(args.examples)}',
        '-B', f'{args.build_dir}/{args.build_type}',
        '-S', '.'
    ])

if __name__ == '__main__':
    sys.exit(main())
