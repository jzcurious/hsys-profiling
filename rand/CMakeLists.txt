cmake_minimum_required(VERSION 3.31)
project(hsys_profiling_rand LANGUAGES CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

find_package(CUDAToolkit REQUIRED)

add_library(rand INTERFACE)

target_compile_features(rand INTERFACE
    cuda_std_20
)

target_compile_options(rand INTERFACE
    -Wall -O3
)

target_include_directories(rand INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_link_libraries(rand INTERFACE
    CUDA::curand
)
