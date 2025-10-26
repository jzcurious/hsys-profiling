cmake_minimum_required(VERSION 3.31)
project(hsys_profiling_core LANGUAGES CUDA)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

add_library(core INTERFACE)

target_compile_features(core INTERFACE
    cuda_std_20
)

target_compile_options(core INTERFACE
    -Wall -O3
)

target_include_directories(core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)
