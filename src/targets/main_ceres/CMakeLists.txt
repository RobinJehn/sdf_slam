# Define the source directory
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Include the source files definitions
include(${SRC_DIR}/source_files.cmake)

# Collect source files for this executable
set(SRC_FILES
    targets/main_ceres/main_ceres.cpp
    ${MAP_SOURCES}
    ${SCAN_SOURCES}
    ${STATE_SOURCES}
    ${OPTIMIZATION_SOURCES_COMMON}
    optimization/objective_ceres.cpp
    ${VISUALIZE_SOURCES}
)

# Prepend SRC_DIR to all source files
list(TRANSFORM SRC_FILES PREPEND "${SRC_DIR}/")

# Create the executable
add_executable(sdf_slam_ceres ${SRC_FILES})

# Include directories
target_include_directories(sdf_slam_ceres PRIVATE
    ${SRC_DIR}
)

# Link external libraries
target_link_libraries(sdf_slam_ceres
    ${PCL_LIBRARIES}
    Eigen3::Eigen
    ${OpenCV_LIBS}
    Ceres::ceres
    glog::glog
)

# Set target properties
set_target_properties(sdf_slam_ceres PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
