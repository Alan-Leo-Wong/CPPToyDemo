cmake_minimum_required(VERSION 3.18)
project(readFile LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(in_obj in_obj.cpp)

find_package(Eigen3 REQUIRED CONFIG)

if(TARGET Eigen3::Eigen)
    message(STATUS "Eigen3 v${EIGEN3_VERSION_STRING} found in ${EIGEN3_INCLUDE_DIR}")
endif()

target_link_libraries(in_obj PUBLIC Eigen3::Eigen)