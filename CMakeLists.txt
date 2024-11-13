cmake_minimum_required(VERSION 3.10)

# Set the project name
project(sdf_slam VERSION 1.0 LANGUAGES CXX)

# Specify the C++ standard globally
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_MAIN "Build main executable" ON)
option(BUILD_MAIN_EIGEN "Build main_eigen executable" OFF)
option(BUILD_MAIN_CERES "Build main_ceres executable" OFF)
option(BUILD_TESTS "Build the tests" OFF)

# Add the directory with FindSuiteSparse.cmake to CMake's module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

# Define common variables for build type
set(CMAKE_BUILD_TYPE Debug)

# ================================================
# Find external libraries
# ================================================
find_package(PCL 1.8 REQUIRED COMPONENTS common io visualization)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(JsonCpp REQUIRED)

include(FetchContent)
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0
)
FetchContent_MakeAvailable(yaml-cpp)

# ================================================
# Add Subdirectories
# ================================================
add_subdirectory(src)

# ================================================
# Add Tests if enabled
# ================================================
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()