cmake_minimum_required(VERSION 3.24)
project(hsys_profiling_pack LANGUAGES CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

add_library(pack INTERFACE)

target_compile_features(pack INTERFACE
    cuda_std_20
)

target_compile_options(pack INTERFACE
    -Wall -O3 --expt-relaxed-constexpr
)

target_include_directories(pack INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_link_libraries(pack INTERFACE
    core
)
