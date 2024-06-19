function(add_example EXAMPLE_NAME)
    add_executable(${EXAMPLE_NAME})
    target_sources(${EXAMPLE_NAME} PRIVATE src/${EXAMPLE_NAME}.cpp)
    target_link_libraries(${EXAMPLE_NAME} PRIVATE spdlog::spdlog opencv::opencv)
    target_compile_features(${EXAMPLE_NAME} PUBLIC cxx_std_20)
    install(TARGETS ${EXAMPLE_NAME} DESTINATION examples)
endfunction()

find_package(spdlog REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(triton-client REQUIRED)

add_example(client_example_model_info)
target_link_libraries(client_example_model_info PRIVATE teiacare::inference_client)

add_example(client_example_server_info)
target_link_libraries(client_example_server_info PRIVATE teiacare::inference_client)

add_example(client_example_simple_inference)
target_link_libraries(client_example_simple_inference PRIVATE teiacare::inference_client)

add_example(client_example_image_classifier)
target_link_libraries(client_example_image_classifier PRIVATE teiacare::inference_client)

add_example(client_example_yolo)
target_link_libraries(client_example_yolo PRIVATE teiacare::inference_client)

# add_example(triton_client_simple_example)
# target_link_libraries(triton_client_simple_example PRIVATE triton-client::triton-client)

# add_example(triton_client_image_inference)
# target_link_libraries(triton_client_image_inference PRIVATE triton-client::triton-client)
