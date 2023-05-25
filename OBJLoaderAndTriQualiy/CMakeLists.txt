cmake_minimum_required(VERSION 3.16)
project(TriQuality)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
include(libigl)

# Enable the target igl::glfw
igl_include(glfw)

find_package(Eigen3 REQUIRED CONFIG)

# Add your project files
add_executable(${PROJECT_NAME} triQuality.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC igl::glfw)

if(TARGET Eigen3::Eigen)
    message(STATUS "Eigen3 v${EIGEN3_VERSION_STRING} found in ${EIGEN3_INCLUDE_DIR}")
    target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
endif()
