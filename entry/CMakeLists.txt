cmake_minimum_required(VERSION 3.24)
project(hsys_profiling_entry LANGUAGES CUDA)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

include(../cmake/setup_cuda_targets.cmake)

find_package(CUDAToolkit REQUIRED)

add_executable(dot ${CMAKE_CURRENT_SOURCE_DIR}/src/dot.cu)

setup_cuda_targets(
    TARGETS
        dot
    FEATURES
        cuda_std_20
    OPTIONS
        -Wall -O3
    CUDA_MIN_ARCH
        75
    LIBS
        kernels
        CUDA::curand
)
