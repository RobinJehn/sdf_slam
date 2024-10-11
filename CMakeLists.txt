cmake_minimum_required(VERSION 3.10)

# Set the project name
project(sdf_slam)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Debug)

# Add the sources
set(MAP_PATH src/map)
set(SCAN_PATH src/scan)
set(STATE_PATH src/state)
set(OPTIMIZATION_PATH src/optimization)

# Source and header files for common components
set(MAP_SOURCES
    ${MAP_PATH}/map.cpp
    ${MAP_PATH}/utils.cpp
)
set(MAP_HEADERS
    ${MAP_PATH}/map.hpp
    ${MAP_PATH}/utils.hpp
)

set(SCAN_SOURCES
    ${SCAN_PATH}/generate.cpp
)
set(SCAN_HEADERS
    ${SCAN_PATH}/generate.hpp
)

set(STATE_SOURCES
    ${STATE_PATH}/state.cpp
)
set(STATE_HEADERS
    ${STATE_PATH}/state.hpp
)

# Common optimization sources (excluding objective)
set(OPTIMIZATION_SOURCES_COMMON
    ${OPTIMIZATION_PATH}/optimizer.cpp
    ${OPTIMIZATION_PATH}/utils.cpp
)
set(OPTIMIZATION_HEADERS_COMMON
    ${OPTIMIZATION_PATH}/optimizer.hpp
    ${OPTIMIZATION_PATH}/utils.hpp
)

# Include directory for source files
include_directories(src)

# Find and include PCL
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Find and include Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find and include OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ================================================
# Target: sdf_slam (Levenberg-Marquardt-based)
# ================================================
set(OPTIMIZATION_SOURCES_LM
    ${OPTIMIZATION_PATH}/objective.cpp
)

set(OPTIMIZATION_HEADERS_LM
    ${OPTIMIZATION_PATH}/objective.hpp
)

# Create executable for sdf_slam (Levenberg-Marquardt)
add_executable(sdf_slam
    src/main.cpp
    ${MAP_SOURCES}
    ${SCAN_SOURCES}
    ${STATE_SOURCES}
    ${OPTIMIZATION_SOURCES_COMMON}
    ${OPTIMIZATION_SOURCES_LM}
)

# Include directories for sdf_slam (after target creation)
target_include_directories(sdf_slam PRIVATE ${EIGEN3_INCLUDE_DIR})

target_link_libraries(sdf_slam
    ${PCL_LIBRARIES}
    Eigen3::Eigen
    ${OpenCV_LIBS}
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

# Find and include Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Find glog
find_package(glog REQUIRED)

# Create executable for sdf_slam_ceres (Ceres-based)
add_executable(sdf_slam_ceres
    src/main_ceres.cpp
    ${MAP_SOURCES}
    ${SCAN_SOURCES}
    ${STATE_SOURCES}
    ${OPTIMIZATION_SOURCES_COMMON}
    ${OPTIMIZATION_SOURCES_CERES}
)

# Include directories for sdf_slam_ceres
target_include_directories(sdf_slam_ceres PRIVATE ${EIGEN3_INCLUDE_DIR})

target_link_libraries(sdf_slam_ceres
    ${PCL_LIBRARIES}
    Eigen3::Eigen
    ${OpenCV_LIBS}
    Ceres::ceres
    glog::glog
)
