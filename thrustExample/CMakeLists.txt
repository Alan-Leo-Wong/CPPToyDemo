cmake_minimum_required(VERSION 3.18)

# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
# message(STATUS "Failed to detect a default CUDA architecture. Use '-DCMAKE_CUDA_ARCHITECTURES=XX' to specify the default value.")
# set(CMAKE_CUDA_ARCHITECTURES 86)
# endif()
project(MarchingCubes LANGUAGES CXX CUDA)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# target_include_directories(MC PRIVATE include)
add_executable(TEST exclusive_scan.cu)
target_compile_features(TEST PRIVATE cuda_std_17)

# target_compile_features(TEST PRIVATE cxx_std_17)