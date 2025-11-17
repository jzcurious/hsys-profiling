cmake_minimum_required(VERSION 3.24)
project(hsys_profiling_bench LANGUAGES CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

set(BENCHMARK_ENABLE_TESTING OFF)

include(FetchContent)

FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.2
)
FetchContent_MakeAvailable(benchmark)

include(../cmake/setup_cuda_targets.cmake)

add_executable(dot_bench ${CMAKE_CURRENT_SOURCE_DIR}/src/dot_bench.cu)
setup_cuda_targets(
    TARGETS
        dot_bench
    FEATURES
        cuda_std_20
    OPTIONS
        -Wall -O3
    CUDA_MIN_ARCH
        75
    LIBS
        dot
        benchmark::benchmark_main
)

add_executable(matmul_bench ${CMAKE_CURRENT_SOURCE_DIR}/src/matmul_bench.cu)
setup_cuda_targets(
    TARGETS
        matmul_bench
    FEATURES
        cuda_std_20
    OPTIONS
        -Wall -O3
    CUDA_MIN_ARCH
        75
    LIBS
        matmul
        benchmark::benchmark_main
)

add_executable(pack_bench ${CMAKE_CURRENT_SOURCE_DIR}/src/pack_bench.cu)
setup_cuda_targets(
    TARGETS
        pack_bench
    FEATURES
        cuda_std_20
    OPTIONS
        -Wall -O3
    CUDA_MIN_ARCH
        75
    LIBS
        pack
        benchmark::benchmark_main
)
