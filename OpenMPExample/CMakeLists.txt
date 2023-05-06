cmake_minimum_required(VERSION 3.18)
project(readFile LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(TEST main.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(TEST PUBLIC OpenMP::OpenMP_CXX)
endif()
