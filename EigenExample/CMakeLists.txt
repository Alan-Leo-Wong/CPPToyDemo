cmake_minimum_required(VERSION 3.18)
project(3Slabs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED CONFIG)

# set(EIGEN3_INCLUDE_DIR "E:\\3rdParty\\eigen-3.4.0")

# add_subdirectory(include)
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    list(APPEND CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
endif(OpenMP_CXX_FOUND)

message(${OpenMP_CXX_FLAGS})

# add_executable(main get_absolute_min.cpp)
# add_executable(main eigenVectorToMatrix.cpp)
# add_executable(main 2Dand3D.cpp)
# add_executable(main sparse_matrix.cpp)
add_executable(main equal_operator.cpp)

# target_include_directories(main PRIVATE include)
if(TARGET Eigen3::Eigen)
    message(STATUS "Eigen3 v${EIGEN3_VERSION_STRING} found in ${EIGEN3_INCLUDE_DIR}")
    target_link_libraries(main Eigen3::Eigen)

    message(STATUS ${EIGEN3_INCLUDE_DIR})

    # include_directories(${EIGEN3_INCLUDE_DIR})

    # link_libraries(Eigen3::Eigen)
endif()
