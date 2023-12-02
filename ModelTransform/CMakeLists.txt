cmake_minimum_required(VERSION 3.18)
project(3Slabs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED CONFIG)

find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    list(APPEND CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
endif(OpenMP_CXX_FOUND)

message(${OpenMP_CXX_FLAGS})

add_executable(main_1 point_translate_zoom.cpp)
add_executable(main_2 point_rotate.cpp)

if(TARGET Eigen3::Eigen)
    message(STATUS "Eigen3 v${EIGEN3_VERSION_STRING} found in ${EIGEN3_INCLUDE_DIR}")
    target_link_libraries(main_1 Eigen3::Eigen)
    target_link_libraries(main_2 Eigen3::Eigen)

    message(STATUS ${EIGEN3_INCLUDE_DIR})
endif()
