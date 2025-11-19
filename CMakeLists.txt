cmake_minimum_required(VERSION 3.5)

project(mapannotator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(YAML-CPP REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

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
    segmentation/downsample_seeding.hpp segmentation/downsample_seeding.cpp
)

set(PDDL_SRC
    pddl/pddlgenerator.h pddl/pddlgenerator.cpp
)

add_executable(${PROJECT_NAME} main.cpp
                               ${ZONEGRAPH_SRC}
                               ${PREPARING_SRC}
                               ${SEGMENTATION_SRC}
                               ${PDDL_SRC}
                               map_processing.cpp
                               utils.hpp config.hpp)

target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} yaml-cpp)

add_executable(mapannotator_ros2 ros2_node.cpp
                               ${ZONEGRAPH_SRC}
                               ${PREPARING_SRC}
                               ${SEGMENTATION_SRC}
                               ${PDDL_SRC}
                               map_processing.cpp
                               utils.hpp config.hpp)
target_link_libraries(mapannotator_ros2
                      ${OpenCV_LIBS} yaml-cpp)

ament_target_dependencies(mapannotator_ros2
                          rclcpp
                          nav_msgs
                          std_msgs
                          sensor_msgs)

install(TARGETS ${PROJECT_NAME} mapannotator_ros2
        RUNTIME DESTINATION lib/${PROJECT_NAME})

ament_package()
