set(TARGET_NAME teiacare_inference_client)
add_library(${TARGET_NAME})
add_library(teiacare::inference_client ALIAS ${TARGET_NAME})

find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)

set(PROTO_FILENAME services)
set(PROTO_FILE ${PROTO_FILENAME}.proto)
set(PROTO_IN_DIR ${PROJECT_SOURCE_DIR}/proto)
set(PROTO_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto)
include(grpc_common)

configure_file(
    src/version.cpp.in
    ${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp
)

set(TARGET_HEADERS
    include/teiacare/inference_client/client_factory.hpp
    include/teiacare/inference_client/client_interface.hpp
    include/teiacare/inference_client/data_type.hpp
    include/teiacare/inference_client/infer_request.hpp
    include/teiacare/inference_client/infer_response.hpp
    include/teiacare/inference_client/infer_tensor.hpp
    include/teiacare/inference_client/model_metadata.hpp
    include/teiacare/inference_client/server_metadata.hpp
    include/teiacare/inference_client/timeout_error.hpp
)

set(TARGET_SOURCES
    src/client_factory.cpp
    src/client_rpc_unary_async.hpp
    src/data_type.cpp
    src/grpc_client.cpp
    src/grpc_client.hpp
    src/tensor_converter.cpp
    src/tensor_converter.hpp
    ${GRPC_PROTO_FILES}
)

target_compile_features(${TARGET_NAME} PUBLIC cxx_std_20)
target_sources(${TARGET_NAME} PUBLIC ${TARGET_HEADERS} PRIVATE ${TARGET_SOURCES})
target_link_libraries(${TARGET_NAME} PRIVATE gRPC::grpc++)
target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<INSTALL_INTERFACE:CMAKE_INSTALL_INCLUDEDIR>
    PRIVATE
        ${PROTO_OUT_DIR}
)
set_target_properties(${TARGET_NAME} PROPERTIES VERSION ${${PROJECT_NAME}_VERSION} SOVERSION ${${PROJECT_NAME}_VERSION_MAJOR})
set_target_properties(${TARGET_NAME} PROPERTIES PUBLIC_HEADER "${TARGET_HEADERS}")
install(TARGETS ${TARGET_NAME} PUBLIC_HEADER DESTINATION include/teiacare/inference_client)

if(TC_ENABLE_WARNINGS_ERROR)
    include(warnings)
    add_warnings(${TARGET_NAME})
    add_warnings_as_errors(${TARGET_NAME})
endif()

#################################################################
# Unit Tests
if(TC_ENABLE_UNIT_TESTS)
    set(TEST_TARGET_NAME teiacare_inference_client_unittests)
    find_package(GTest REQUIRED)
    include(GoogleTest)

    set(UNIT_TESTS_SRC
        tests/main.cpp
        tests/test_client_factory.cpp
    )

    add_executable(${TEST_TARGET_NAME})
    target_sources(${TEST_TARGET_NAME} PRIVATE ${UNIT_TESTS_SRC})
    target_include_directories(${TEST_TARGET_NAME} PRIVATE  ${CMAKE_CURRENT_BINARY_DIR}/proto)
    target_compile_features(${TEST_TARGET_NAME} PUBLIC cxx_std_20)
    target_link_libraries(${TEST_TARGET_NAME}
        PRIVATE
            GTest::gtest
            GTest::gmock
            teiacare::inference_client
            gRPC::grpc++
    )

    gtest_discover_tests(${TEST_TARGET_NAME})
    install(TARGETS ${TEST_TARGET_NAME} DESTINATION unit_tests)

    if(TC_ENABLE_WARNINGS_ERROR)
        add_warnings(${TEST_TARGET_NAME})
        add_warnings_as_errors(${TEST_TARGET_NAME})
    endif()

    if(TC_ENABLE_UNIT_TESTS_COVERAGE)
        include(coverage)
        add_coverage(${TARGET_NAME})
    endif()
endif()

#################################################################
# Examples
if(TC_ENABLE_EXAMPLES)
include(examples)
    # add_example(teiacare::inference_client client_example_image_classifier)
    add_example(teiacare::inference_client client_example_model_info)
    add_example(teiacare::inference_client client_example_server_info)
    add_example(teiacare::inference_client client_example_simple_inference)
endif()
