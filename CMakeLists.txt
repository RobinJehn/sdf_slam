cmake_minimum_required(VERSION 3.10)

# Set the project name
project(sdf_slam)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Debug)

# Add the sources
set(MAP_PATH
    src/map
)
set(MAP_SOURCES
    ${MAP_PATH}/map.cpp
    ${MAP_PATH}/utils.cpp
)
set(MAP_HEADERS
    ${MAP_PATH}/map.hpp
    ${MAP_PATH}/utils.hpp
)

set(SCAN_PATH
    src/scan
)

set(SCAN_SOURCES
    ${SCAN_PATH}/generate.cpp
)
set(SCAN_HEADERS
    ${SCAN_PATH}/generate.hpp
)

set(STATE_PATH
    src/state
)
set(STATE_SOURCES
    ${STATE_PATH}/state.cpp)
set(STATE_HEADERS
    ${STATE_PATH}/state.hpp
)
set(OPTIMIZATION_PATH
    src/optimization
)
set(OPTIMIZATION_SOURCES
    ${OPTIMIZATION_PATH}/objective.cpp
    ${OPTIMIZATION_PATH}/optimizer.cpp
    ${OPTIMIZATION_PATH}/utils.cpp
)
set(OPTIMIZATION_HEADERS
    ${OPTIMIZATION_PATH}/objective.hpp
    ${OPTIMIZATION_PATH}/optimizer.hpp
    ${OPTIMIZATION_PATH}/utils.hpp
)

set(SOURCES
    src/main.cpp
    ${MAP_SOURCES}
    ${STATE_SOURCES}
    ${OPTIMIZATION_SOURCES}
    ${SCAN_SOURCES}
)

set(HEADERS
    ${MAP_HEADERS}
    ${STATE_HEADERS}
    ${OPTIMIZATION_HEADERS}
    ${SCAN_HEADERS}
)

# Add the executable
add_executable(sdf_slam ${SOURCES} ${HEADERS})

# Find PCL
find_package(PCL 1.8 REQUIRED)

# Include PCL directories
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Link PCL libraries
target_link_libraries(sdf_slam ${PCL_LIBRARIES})

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include Eigen3 directories
target_include_directories(sdf_slam PRIVATE ${EIGEN3_INCLUDE_DIR})

# Link Eigen3 libraries
target_link_libraries(sdf_slam Eigen3::Eigen)

# Include src directory
target_include_directories(sdf_slam PRIVATE src)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Link OpenCV libraries
target_link_libraries(sdf_slam ${OpenCV_LIBS})
