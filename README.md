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
