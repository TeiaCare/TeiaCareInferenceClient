cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(teiacare_inference_client_test_package CXX)

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build/modules)
find_package(teiacare_inference_client CONFIG REQUIRED)

add_executable(teiacare_inference_client_test_package src/test.cpp)
target_link_libraries(teiacare_inference_client_test_package PRIVATE teiacare::inference_client)
install(TARGETS teiacare_inference_client_test_package DESTINATION .)
