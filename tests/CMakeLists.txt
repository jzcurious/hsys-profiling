cmake_minimum_required(VERSION 3.24)
project(hsys_profiling_tests LANGUAGES CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

include(FetchContent)
include(GoogleTest)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(tests ${SRC})

target_compile_features(tests PRIVATE
    cuda_std_20
)

target_compile_options(tests PRIVATE
    -Wall -O3
)

target_link_libraries(tests PRIVATE
    core
    vadd
    rand
    GTest::gtest_main
)

enable_testing()
gtest_discover_tests(tests)
