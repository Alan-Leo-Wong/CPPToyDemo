cmake_minimum_required(VERSION 3.18)

# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
# message(STATUS "Failed to detect a default CUDA architecture. Use '-DCMAKE_CUDA_ARCHITECTURES=XX' to specify the default value.")
# set(CMAKE_CUDA_ARCHITECTURES 86)
# endif()
project(THRUST_EXAMPLE LANGUAGES CXX CUDA)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(Eigen3 REQUIRED CONFIG)

# if(TARGET Eigen3::Eigen)
#     message(STATUS "Eigen3 v${EIGEN3_VERSION_STRING} found in ${EIGEN3_INCLUDE_DIR}")
# endif()

# target_include_directories(MC PRIVATE include)
add_executable(TEST axsum.cu)
target_compile_features(TEST PRIVATE cuda_std_17)
# target_link_libraries(TEST PUBLIC Eigen3::Eigen)

# target_compile_features(TEST PRIVATE cxx_std_17)