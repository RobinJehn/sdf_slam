# Add subdirectories for executables
if(BUILD_MAIN)
    add_subdirectory(main)
endif()

if(BUILD_MAIN_INCREMENTAL)
    add_subdirectory(main_incremental)
endif()

if(BUILD_MAIN_EIGEN)
    add_subdirectory(main_eigen)
endif()

if(BUILD_MAIN_CERES)
    add_subdirectory(main_ceres)
endif()

if(BUILD_GENERATE_SCAN)
    add_subdirectory(generate_scan)
endif()
