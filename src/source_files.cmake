# Paths to source files in subfolders
set(MAP_SOURCES
    map/map.cpp
    map/utils.cpp
    map/map_derivatives_2d.cpp
    map/map_derivatives_3d.cpp
)

set(SCAN_SOURCES
    scan/shape.cpp
    scan/scene.cpp
    scan/scan.cpp
)

set(STATE_SOURCES
    state/state.cpp
)

set(OPTIMIZATION_SOURCES_COMMON
    optimization/utils.cpp
    optimization/residuals.cpp
    optimization/residuals_2d.cpp
    optimization/residuals_3d.cpp
    optimization/derivatives.cpp
)

set(VISUALIZE_SOURCES
    visualize/utils.cpp
)

set(CONFIG_SOURCES
    config/utils.cpp
)
