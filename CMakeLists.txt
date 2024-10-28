cmake_minimum_required(VERSION 3.10)

# Set the project name
project(sdf_slam VERSION 1.0 LANGUAGES CXX)

# Specify the C++ standard globally
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Option to build tests
option(BUILD_TESTS "Build the tests" ON)

# Add the directory with FindSuiteSparse.cmake to CMake's module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

# Define common variables for build type and paths
set(CMAKE_BUILD_TYPE Debug)
set(MAP_PATH src/map)
set(SCAN_PATH src/scan)
set(STATE_PATH src/state)
set(OPTIMIZATION_PATH src/optimization)
set(VISUALIZE_PATH src/visualize)

# Add common sources and headers
set(MAP_SOURCES
    ${MAP_PATH}/map.cpp
    ${MAP_PATH}/utils.cpp
)
set(SCAN_SOURCES
    ${SCAN_PATH}/generate.cpp
)
set(STATE_SOURCES
    ${STATE_PATH}/state.cpp
)
set(OPTIMIZATION_SOURCES_COMMON
    ${OPTIMIZATION_PATH}/utils.cpp
)
set(VISUALIZE_SOURCES
    ${VISUALIZE_PATH}/utils.cpp
)

set(MAP_HEADERS
    ${MAP_PATH}/map.hpp
    ${MAP_PATH}/utils.hpp
)
set(SCAN_HEADERS
    ${SCAN_PATH}/generate.hpp
)
set(STATE_HEADERS
    ${STATE_PATH}/state.hpp
)
set(OPTIMIZATION_HEADERS_COMMON
    ${OPTIMIZATION_PATH}/utils.hpp
)
set(VISUALIZE_HEADERS
    ${VISUALIZE_PATH}/utils.hpp
)

# ================================================
# Find and include external libraries
# ================================================

# Find and include PCL
find_package(PCL 1.8 REQUIRED COMPONENTS common io visualization)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(JsonCpp REQUIRED)

# ================================================
# Target: sdf_slam_eigen
# ================================================
set(OPTIMIZATION_SOURCES_EIGEN
    ${OPTIMIZATION_PATH}/optimizer.cpp
    ${OPTIMIZATION_PATH}/objective.cpp
)
set(OPTIMIZATION_HEADERS_EIGEN
    ${OPTIMIZATION_PATH}/optimizer.hpp
    ${OPTIMIZATION_PATH}/objective.hpp
)

# Create executable for sdf_slam_eigen
add_executable(${PROJECT_NAME}_eigen
    src/main_eigen.cpp
    ${MAP_SOURCES}
    ${SCAN_SOURCES}
    ${STATE_SOURCES}
    ${OPTIMIZATION_SOURCES_COMMON}
    ${OPTIMIZATION_SOURCES_EIGEN}
    ${VISUALIZE_SOURCES}
)

# Link libraries and include directories for sdf_slam_eigen
target_link_libraries(${PROJECT_NAME}_eigen
    ${PCL_LIBRARIES}
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

target_include_directories(${PROJECT_NAME}_eigen PRIVATE
    src
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# ================================================
# Target: sdf_slam_ceres (Ceres-based)
# ================================================
set(OPTIMIZATION_SOURCES_CERES
    ${OPTIMIZATION_PATH}/objective_ceres.cpp
)

set(OPTIMIZATION_HEADERS_CERES
    ${OPTIMIZATION_PATH}/objective_ceres.hpp
)

# Create executable for sdf_slam_ceres (Ceres-based)
add_executable(${PROJECT_NAME}_ceres
    src/main_ceres.cpp
    ${MAP_SOURCES}
    ${SCAN_SOURCES}
    ${STATE_SOURCES}
    ${OPTIMIZATION_SOURCES_COMMON}
    ${OPTIMIZATION_SOURCES_CERES}
    ${VISUALIZE_SOURCES}
)

# Link libraries and include directories for sdf_slam_ceres
target_link_libraries(${PROJECT_NAME}_ceres
    ${PCL_LIBRARIES}
    Eigen3::Eigen
    ${OpenCV_LIBS}
    Ceres::ceres
    glog::glog
)

target_include_directories(${PROJECT_NAME}_ceres PRIVATE
    src
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

# ================================================
# Target: sdf_slam
# ================================================
set(OPTIMIZATION_SOURCES_LM
    ${OPTIMIZATION_PATH}/optimizer.cpp
    ${OPTIMIZATION_PATH}/objective.cpp
)
set(OPTIMIZATION_HEADERS_LM
    ${OPTIMIZATION_PATH}/optimizer.hpp
    ${OPTIMIZATION_PATH}/objective.hpp
)

# Create executable for sdf_slam
add_executable(${PROJECT_NAME}
    src/main.cpp
    ${MAP_SOURCES}
    ${SCAN_SOURCES}
    ${STATE_SOURCES}
    ${OPTIMIZATION_SOURCES_COMMON}
    ${OPTIMIZATION_SOURCES_LM}
    ${VISUALIZE_SOURCES}
)

# Link libraries and include directories for sdf_slam
target_link_libraries(${PROJECT_NAME}
    ${PCL_LIBRARIES}
    Eigen3::Eigen
    ${OpenCV_LIBS}
    SuiteSparse::CHOLMOD # Add SuiteSparse CHOLMOD
    SuiteSparse::SPQR # Add SPQR for SuiteSparseQR functionality
)

target_include_directories(${PROJECT_NAME} PRIVATE
    src
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# ================================================
# Add Tests if enabled
# ================================================
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
