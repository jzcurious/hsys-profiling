cmake_minimum_required(VERSION 3.24)
project(hsys_profiling_entry LANGUAGES CUDA)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

include(../cmake/setup_cuda_targets.cmake)

find_package(CUDAToolkit REQUIRED)

add_executable(dot_suite ${CMAKE_CURRENT_SOURCE_DIR}/src/dot.cu)

setup_cuda_targets(
    TARGETS
        dot_suite
    FEATURES
        cuda_std_20
    OPTIONS
        -Wall -O3
    CUDA_MIN_ARCH
        75
    LIBS
        dot
        CUDA::curand
)

add_executable(matmul_suite ${CMAKE_CURRENT_SOURCE_DIR}/src/matmul.cu)

setup_cuda_targets(
    TARGETS
        matmul_suite
    FEATURES
        cuda_std_20
    OPTIONS
        -Wall -O3
    CUDA_MIN_ARCH
        75
    LIBS
        matmul
        CUDA::curand
)
