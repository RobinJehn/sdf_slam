# Define the source directory
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Include the source files definitions
include(${SRC_DIR}/source_files.cmake)

# Collect source files for this executable
set(SRC_FILES
    targets/main/main.cpp
    ${MAP_SOURCES}
    ${SCAN_SOURCES}
    ${STATE_SOURCES}
    ${OPTIMIZATION_SOURCES_COMMON}
    optimization/optimizer.cpp
    optimization/objective.cpp
    ${VISUALIZE_SOURCES}
    ${CONFIG_SOURCES}
)

# Prepend SRC_DIR to all source files
list(TRANSFORM SRC_FILES PREPEND "${SRC_DIR}/")

# Create the executable
add_executable(sdf_slam ${SRC_FILES})

# Include directories
target_include_directories(sdf_slam PRIVATE
    ${SRC_DIR}
)

# Link external libraries
target_link_libraries(sdf_slam
    ${PCL_LIBRARIES}
    Eigen3::Eigen
    ${OpenCV_LIBS}
    SuiteSparse::CHOLMOD
    SuiteSparse::SPQR
    yaml-cpp
)

# Set target properties
set_target_properties(sdf_slam PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
