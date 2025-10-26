cmake_minimum_required(VERSION 3.24)
project(hsys_profiling_kernels LANGUAGES CUDA)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

add_library(kernels INTERFACE)

target_compile_features(kernels INTERFACE
    cuda_std_20
)

target_compile_options(kernels INTERFACE
    -Wall -O3
)

target_include_directories(kernels INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_link_libraries(kernels INTERFACE
    core
)
