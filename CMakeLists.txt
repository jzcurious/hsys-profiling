cmake_minimum_required(VERSION 3.24)
project(hsys_profiling LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_subdirectory(core)
add_subdirectory(dot)
add_subdirectory(matmul)
add_subdirectory(entry)
add_subdirectory(bench)
