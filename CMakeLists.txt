cmake_minimum_required(VERSION 3.5)

project(mapannotator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(YAML-CPP REQUIRED)

set(ZONEGRAPH_SRC
    mapgraph/zonegraph.hpp mapgraph/zonegraph.cpp
    mapgraph/zoneconfig.hpp
    mapgraph/zone_graph_dot.hpp mapgraph/zone_graph_draw.hpp
    mapgraph/typeregistry.h mapgraph/typeregistry.cpp
)

set(PREPARING_SRC
    preparing/mappreprocessing.hpp preparing/mappreprocessing.cpp
)

set(SEGMENTATION_SRC
    segmentation/endpoints.h
    segmentation/segmentation.hpp segmentation/segmentation.cpp
    segmentation/labelmapping.hpp segmentation/labelmapping.cpp
    segmentation/zoneclassifier.h segmentation/zoneclassifier.cpp
)

set(PDDL_SRC
    pddl/pddlgenerator.h pddl/pddlgenerator.cpp
)

add_executable(${PROJECT_NAME} main.cpp
                               ${ZONEGRAPH_SRC}
                               ${PREPARING_SRC}
                               ${SEGMENTATION_SRC}
                               ${PDDL_SRC}
                               utils.hpp config.hpp)

target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} yaml-cpp)
