cmake_minimum_required(VERSION 3.20.0)
project(TestCase)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/compiled")
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/dispatcher)
set(MY_SOURCES
        src/tensor.cpp 
        src/TensorFactory.cpp
        src/TensorUtils.cpp
        src/registry/unary_registry.cpp 
        src/registry/unary_dispatcher.cpp
        src/kernels_cpu/unaryops/unified_unary_api.cpp
        src/kernels_cpu/unaryops/exp_log_test4.cpp
        tests/dispatcher/unary/test_exp_ops.cpp
    )
add_executable(testcase ${MY_SOURCES})