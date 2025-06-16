# Map Annotator

A prototype tool for segmenting indoor maps and generating a zone graph. The
project can also produce PDDL descriptions of the resulting graph for robotic
planning experiments.

## Dependencies

The application depends on [OpenCV](https://opencv.org/) and
[yaml-cpp](https://github.com/jbeder/yaml-cpp). On Debian/Ubuntu systems they can
be installed with:

```bash
sudo apt-get install libopencv-dev libyaml-cpp-dev
```

## Building

```bash
mkdir build
cd build
cmake ..
make
```

If CMake cannot find OpenCV or yaml-cpp make sure the development packages are
installed and that the libraries are discoverable via `CMAKE_PREFIX_PATH` or the
`OpenCV_DIR`/`YAML_CPP_DIR` variables.

## Usage

```
./mapannotator <map.pgm> [config.yaml]
```

The program expects a grayscale map image in `.pgm` format and an optional YAML
configuration. It outputs diagnostic information, generates a PDDL problem file
and creates visual previews of the computed zones.

## Repository structure

- `segmentation/` – image processing and segmentation routines
- `mapgraph/` – zone graph data structures and visualisation helpers
- `pddl/` – utilities for generating PDDL from the graph
- `config/` – example classification rules

## License

This project is distributed under the terms of the MIT license. See
[LICENSE](LICENSE) for details.

## ROS 2 Integration

When ROS 2 and the required message packages are available the project can be
built with an additional node that consumes a `nav_msgs/OccupancyGrid` and
publishes the generated PDDL.

The node subscribes to `/map` by default and publishes to `/pddl/map`. Example
build and run steps:

```bash
mkdir build && cd build
cmake ..
make mapannotator_ros2
ros2 run mapannotator mapannotator_ros2
```
