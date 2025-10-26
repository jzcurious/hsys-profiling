cmake_minimum_required(VERSION 3.24)
project(hsys_profiling_entry LANGUAGES CUDA)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(entry ${SRC})

target_compile_features(entry PRIVATE
    cuda_std_20
)

target_compile_options(entry PRIVATE
    -Wall -O3
)

target_link_libraries(entry PRIVATE
    kernels
)
