cmake_minimum_required(VERSION 3.24)
project(hsys_profiling_dot LANGUAGES CUDA)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

add_library(dot INTERFACE)

target_compile_features(dot INTERFACE
    cuda_std_20
)

target_compile_options(dot INTERFACE
    -Wall -O3
)

target_include_directories(dot INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_link_libraries(dot INTERFACE
    core
)
