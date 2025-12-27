cmake_minimum_required(VERSION 3.5)

project(mapannotator LANGUAGES C CXX)

option(MAPANNOTATOR_BUILD_ROS2 "Build ROS 2 node ${PROJECT_NAME}_ros2" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(YAML-CPP REQUIRED)

if(MAPANNOTATOR_BUILD_ROS2)
    find_package(ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(nav_msgs REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(sensor_msgs REQUIRED)
endif()

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

add_executable(${PROJECT_NAME}_cli main.cpp
                               ${ZONEGRAPH_SRC}
                               ${PREPARING_SRC}
                               ${SEGMENTATION_SRC}
                               ${PDDL_SRC}
                               map_processing.cpp
                               utils.hpp config.hpp)

set_target_properties(${PROJECT_NAME}_cli PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}_cli PUBLIC ${OpenCV_LIBS} yaml-cpp)

if(MAPANNOTATOR_BUILD_ROS2)
    add_executable(${PROJECT_NAME}_ros2 ros2_node.cpp
                                   ${ZONEGRAPH_SRC}
                                   ${PREPARING_SRC}
                                   ${SEGMENTATION_SRC}
                                   ${PDDL_SRC}
                                   map_processing.cpp
                                   utils.hpp config.hpp)
    target_link_libraries(${PROJECT_NAME}_ros2
                          ${OpenCV_LIBS} yaml-cpp)

    ament_target_dependencies(${PROJECT_NAME}_ros2
                              rclcpp
                              nav_msgs
                              std_msgs
                              sensor_msgs)

    install(TARGETS ${PROJECT_NAME}_ros2
            RUNTIME DESTINATION lib/${PROJECT_NAME})

    ament_package()
endif()

install(TARGETS ${PROJECT_NAME}_cli
        RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config
        DESTINATION share/${PROJECT_NAME})
