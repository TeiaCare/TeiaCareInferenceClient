find_package(benchmark CONFIG REQUIRED)
# find_package(triton-client REQUIRED)

function(add_timings TIMINGS_TARGET)
    add_executable(${TIMINGS_TARGET})
    target_sources(${TIMINGS_TARGET} PRIVATE src/${TIMINGS_TARGET}.cpp)
    target_compile_features(${TIMINGS_TARGET} PUBLIC cxx_std_20)
    install(TARGETS ${TIMINGS_TARGET} DESTINATION benchmarks)
endfunction()
    
function(add_benchmark BENCHMARK_TARGET)
    add_executable(${BENCHMARK_TARGET})
    target_sources(${BENCHMARK_TARGET} PRIVATE src/${BENCHMARK_TARGET}.cpp)
    target_compile_features(${BENCHMARK_TARGET} PUBLIC cxx_std_20)
    target_link_libraries(${BENCHMARK_TARGET} PRIVATE benchmark::benchmark)
    install(TARGETS ${BENCHMARK_TARGET} DESTINATION benchmarks)
endfunction()

add_timings(timings_teiacare_client)
target_link_libraries(timings_teiacare_client PRIVATE teiacare::inference_client)

add_benchmark(benchmark_teiacare_client)
target_link_libraries(benchmark_teiacare_client PRIVATE teiacare::inference_client)

# add_timings(timings_triton_client)
# target_link_libraries(timings_triton_client PRIVATE triton-client::triton-client)

# add_benchmark(benchmark_triton_client)
# target_link_libraries(benchmark_triton_client PRIVATE triton-client::triton-client)
