#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include "yaml-cpp/yaml.h"

// #define SHOW_DEBUG_IMAGES

#include "mapgraph/zonegraph.hpp"
#include "mapgraph/zone_graph_dot.hpp"
#include "mapgraph/zone_graph_draw.hpp"
#include "preparing/mappreprocessing.hpp"
#include "segmentation/segmentation.hpp"
#include "segmentation/labelmapping.hpp"
#include "segmentation/endpoints.h"
#include "segmentation/zoneclassifier.h"
#include "pddl/pddlgenerator.h"
#include "config.hpp"
#include "utils.hpp"
#include "map_processing.hpp"
#include "visualization.hpp"

using namespace mapping;

int main(int argc, char** argv)
{
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <map.pgm> [config.yaml]" << std::endl;
        return 1;
    }

    SegmenterConfig segmenterConfig;

    std::string pgmFile = argv[1];
    // Если передан файл конфигурации, используем его, иначе "default.yml"
    std::string configFile = (argc >= 3) ? argv[2] : "default.yml";

    // Загружаем YAML-конфигурацию
    YAML::Node config;
    try {
        config = YAML::LoadFile(configFile);
    } catch(const std::exception &e) {
        std::cerr << "Failed to load config file: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Loaded config from: " << configFile << std::endl;

    // Определяем имя map.yaml: то же, что и pgmFile, только с расширением .yaml
    std::string mapYamlFile = pgmFile.substr(0, pgmFile.find_last_of('.')) + ".yaml";
    bool mapYamlLoaded = false;
    std::vector<double> map_origin; // ожидается вектор из 3 элементов: [x, y, theta]
    double resolution = 1.0;         // значение по умолчанию (метров на пиксель)

    if (std::filesystem::exists(mapYamlFile)) {
        try {
            YAML::Node map_yaml = YAML::LoadFile(mapYamlFile);
            std::cout << "Loaded map YAML from: " << mapYamlFile << std::endl;
            if (map_yaml["origin"]) {
                map_origin = map_yaml["origin"].as<std::vector<double>>();
                if(map_origin.size() < 3) {
                    std::cerr << "Map YAML 'origin' field has insufficient elements (expected at least 3)." << std::endl;
                } else {
                    std::cout << "Map origin: x = " << map_origin[0]
                              << ", y = " << map_origin[1]
                              << ", theta = " << map_origin[2] << std::endl;
                }
            } else {
                std::cout << "No origin information in map YAML; origin remains unchanged." << std::endl;
            }
            if (map_yaml["resolution"]) {
                resolution = map_yaml["resolution"].as<double>();
                std::cout << "Map resolution: " << resolution << " meters/pixel" << std::endl;
            } else {
                std::cout << "No resolution information in map YAML; using default: " << resolution << std::endl;
            }
            mapYamlLoaded = true;
        } catch (const std::exception &e) {
            std::cerr << "Error loading map YAML: " << e.what() << std::endl;
        }
    } else {
        std::cout << "Map YAML file not found (" << mapYamlFile << "); origin remains unchanged." << std::endl;
    }

    // Если не удалось загрузить map.yaml, используем значения по умолчанию для origin
    if(!mapYamlLoaded || map_origin.size() < 3) {
        map_origin = {0.0, 0.0, 0.0};
    }

    // Загружаем карту (PGM) в оттенках серого
    cv::Mat raw = cv::imread(pgmFile, cv::IMREAD_GRAYSCALE);
    if(raw.empty()){
        std::cerr << "Failed to load image from: " << pgmFile << std::endl;
        return 1;
    }
    std::cout << "Loaded image: " << pgmFile << " (size: " << raw.cols << "x" << raw.rows << ")" << std::endl;

    showMatDebug("Raw Map", raw);

    // Заполнение структуры MapInfo
    mapInfo.resolution = resolution;
    mapInfo.originX = map_origin[0];
    mapInfo.originY = map_origin[1];
    mapInfo.theta = map_origin[2];
    mapInfo.height = raw.rows;
    mapInfo.width = raw.cols;

    cv::Mat raw8u;
    raw.convertTo(raw8u, CV_8UC1);

    // выравнивание карты(поворот)
    cv::Mat aligned;
    MapPreprocessing::mapAlign(raw8u, aligned, segmenterConfig.alignmentConfig);
    if (aligned.empty())
        aligned = raw8u.clone();
    showMatDebug("aligned Map", aligned);

    // Этап денойзинга
    cv::Mat out;
    auto [rank, cropInfo] = MapPreprocessing::generateDenoisedAlone(aligned, segmenterConfig.denoiseConfig);
    rank.convertTo(out, CV_8U, 255);
    showMatDebug("Denoised Map", out);

    // Расширенние черных зон(препятствия)
    cv::Mat1b binaryDilated = erodeBinary(rank,
                                          segmenterConfig.dilateConfig.kernelSize,
                                          segmenterConfig.dilateConfig.iterations);
    rank.convertTo(out, CV_8U, 255);
    showMatDebug("Dilated Map", out);

    // Получение маски стен
    cv::Mat1b wallMask;
    cv::compare(binaryDilated, 0, wallMask, cv::CMP_EQ); // 255 там, где стена

    // Настройка параметров сегментации
    SegmentationParams segParams;
    segParams.legacyMaxIter = 50;
    segParams.legacySigmaStep = 0.5;
    segParams.legacyThreshold = 0.5;
    segParams.useDownsampleSeeds = true;
    segParams.downsampleConfig.maxIter = 60;
    segParams.downsampleConfig.sigmaStart = 1.0;
    segParams.downsampleConfig.sigmaStep = 0.4;
    segParams.downsampleConfig.threshold = 0.55;

    // Этап сегментации
    LabelsInfo labels;
    auto zones = segmentByGaussianThreshold(binaryDilated, labels, segParams);
    // TODO: сделать проверку если label рядом с препятствием на <= заданному расстоянию то удалить(половина размера робота)

    // Создание матрицы зон
    cv::Mat1i segmentation = cv::Mat::zeros(binaryDilated.size(), CV_32S);
    for (const auto &z : zones) {
        segmentation.setTo(z.label, z.mask);
    }

    // Генерация графа связности зон
    ZoneGraph graph;
    buildGraph(graph, zones, segmentation, mapInfo, labels.centroids);

    // Отрисовка графа связности поверх оригинальной карты
    cv::Mat vis = renderZonesOverlay(zones, raw8u, cropInfo, 0.65);
    mapping::drawZoneGraphOnMap(graph, vis, mapInfo);

    cv::imwrite("segmentation_overlay.png", vis);
    if(!isHeadlessMode())
    {
        cv::imshow("segmented", vis);
    }

    PDDLGenerator gen(graph);
    std::cerr << "\(define \(problem PROBLEM_NAME)\n"
              << "  \(:domain DOMAIN_NAME)\n"
              << gen.objects()
              << gen.init("ROBOT_CUR_ZONE")
              << gen.goal("ROBOT_GOAL_ZONE")
              << ")\n";

//    std::ofstream pddlOut("problem_auto.pddl");
    //    out << "(define (problem demo)\n";
    //    out << "  (:domain floorplan_navigation)\n";
    //    out << gen.objects();
    //    out << gen.init("zone_1");
    //    out << gen.goal("zone_15");
    //    out << ")\n";

    if(!isHeadlessMode())
    {
        std::cout << "Press any key to exit..." << std::endl;
        cv::waitKey(0);
    }
}
