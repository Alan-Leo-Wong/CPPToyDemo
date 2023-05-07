cmake_minimum_required(VERSION 3.18)

# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
# message(STATUS "Failed to detect a default CUDA architecture. Use '-DCMAKE_CUDA_ARCHITECTURES=XX' to specify the default value.")
# set(CMAKE_CUDA_ARCHITECTURES 86)
# endif()
project(THRUST_EXAMPLE LANGUAGES CXX CUDA)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(TEST main.cu)
set_target_properties(TEST
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_compile_features(TEST PRIVATE cxx_std_17 cuda_std_17)

target_link_libraries(TEST PRIVATE CUDA::cublas) # cublas has no static targets on windows.

file(GLOB THIN_SHELLS_SRCS FALSE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    *.cpp
    *.cu
    *.h
    *.hpp)

# aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} THIN_SHELLS_SRCS)
set(THIN_SHELLS_SRC)

foreach(_source IN LISTS THIN_SHELLS_SRCS)
    message(STATUS "src = ${_source}")

    if(NOT ${_source} STREQUAL "main.cu")
        list(APPEND THIN_SHELLS_SRC ${_source})
    endif()
endforeach(_source IN LISTS THIN_SHELLS_SRCS)
