cmake_minimum_required(VERSION 3.15)

# Project setup
project(test_deriv)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find and include Eigen and PCL
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PCL 1.8 REQUIRED)

# Download and unpack GoogleTest at configure time
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# Enable GoogleTest framework
enable_testing()

# Set GoogleTest's libraries correctly
set(GTEST_LIBS gtest gtest_main)

# Add the source files that are necessary for testing
set(OPTIMIZATION_SOURCES
    ${CMAKE_SOURCE_DIR}/src/optimization/objective.cpp
    ${CMAKE_SOURCE_DIR}/src/optimization/optimizer.cpp
    ${CMAKE_SOURCE_DIR}/src/optimization/utils.cpp
    ${CMAKE_SOURCE_DIR}/src/map/map.cpp
    ${CMAKE_SOURCE_DIR}/src/map/utils.cpp
    ${CMAKE_SOURCE_DIR}/src/state/state.cpp
)

# Define the executable for the test
add_executable(${PROJECT_NAME} test_deriv.cpp ${OPTIMIZATION_SOURCES})

# Include directories for the test
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
)

# Link libraries to the test target
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        ${GTEST_LIBS}        # GoogleTest libraries
        Eigen3::Eigen        # Eigen library
        ${PCL_LIBRARIES}     # PCL libraries
)

# Discover and register GoogleTest tests
include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})
