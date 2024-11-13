# Define the source directory
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Include the source files definitions
include(${SRC_DIR}/source_files.cmake)

# Collect source files specific to this executable
set(SRC_FILES
    targets/generate_scan/main.cpp
    ${MAP_SOURCES}
    ${SCAN_SOURCES}
)

# Prepend SRC_DIR to all source files
list(TRANSFORM SRC_FILES PREPEND "${SRC_DIR}/")

# Print the source files for debugging purposes
message(STATUS "Source files: ${SRC_FILES}")

# Create the executable
add_executable(generate_scans ${SRC_FILES})

# Include directories
target_include_directories(generate_scans PRIVATE
    ${SRC_DIR}
)

# Link external libraries
target_link_libraries(generate_scans
    ${PCL_LIBRARIES}
    Eigen3::Eigen
    ${OpenCV_LIBS}
    yaml-cpp
)

# Set target properties
set_target_properties(generate_scans PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
