#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include "yaml-cpp/yaml.h"

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

    double seedClearanceMeters = 0.0;
    double seedClearancePx = 0.0;

    // Загружаем YAML-конфигурацию
    YAML::Node config;
    try {
        config = YAML::LoadFile(configFile);
    } catch(const std::exception &e) {
        std::cerr << "Failed to load config file: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Loaded config from: " << configFile << std::endl;

    if (config["segmentation"])
    {
        auto segNode = config["segmentation"];
        if (segNode["seed_clearance_px"])
            seedClearancePx = segNode["seed_clearance_px"].as<double>();
        if (segNode["seed_clearance_m"])
            seedClearanceMeters = segNode["seed_clearance_m"].as<double>();
    }
    if (seedClearanceMeters <= 0.0 && config["robot"])
    {
        auto robotNode = config["robot"];
        if (robotNode["radius"])
            seedClearanceMeters = robotNode["radius"].as<double>();
        else if (robotNode["diameter"])
            seedClearanceMeters = 0.5 * robotNode["diameter"].as<double>();
    }

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

    if (seedClearancePx <= 0.0 && seedClearanceMeters > 0.0 && mapInfo.resolution > 0.0)
        seedClearancePx = seedClearanceMeters / mapInfo.resolution;

    cv::Mat raw8u;
    raw.convertTo(raw8u, CV_8UC1);

    // выравнивание карты(поворот)
    cv::Mat aligned;
    double alignmentAngle = MapPreprocessing::mapAlign(raw8u,
                                                       aligned,
                                                       segmenterConfig.alignmentConfig);
    mapInfo.theta += alignmentAngle;

    if (aligned.empty())
        aligned = raw8u.clone();
    showMatDebug("aligned Map", aligned);

    // Этап денойзинга
    cv::Mat out;
    auto [rank, cropInfo] = MapPreprocessing::generateDenoisedAlone(aligned, segmenterConfig.denoiseConfig);
    {
        // Crop shifts the origin by removed left/bottom margins; apply to mapInfo.
        int newWidth  = aligned.cols - cropInfo.left - cropInfo.right;
        int newHeight = aligned.rows - cropInfo.top  - cropInfo.bottom;
        if (newWidth > 0 && newHeight > 0) {
            double dx = cropInfo.left   * mapInfo.resolution;
            double dy = cropInfo.bottom * mapInfo.resolution;
            double c  = std::cos(mapInfo.theta);
            double s  = std::sin(mapInfo.theta);
            mapInfo.originX += dx * c - dy * s;
            mapInfo.originY += dx * s + dy * c;
            mapInfo.width  = rank.cols;
            mapInfo.height = rank.rows;
        }
    }
    rank.convertTo(out, CV_8U, 255);
    showMatDebug("Denoised Map", out);

    // Расширенние черных зон(препятствия)
    // TODO: нужно ли это расширение стен?
    cv::Mat1b binaryDilated = erodeBinary(rank,
                                          segmenterConfig.dilateConfig.kernelSize,
                                          segmenterConfig.dilateConfig.iterations);
    rank.convertTo(out, CV_8U, 255);
    showMatDebug("Dilated Map", out);

    // Настройка параметров сегментации
    SegmentationParams segParams;
    segParams.maxIter = 50;
    segParams.sigmaStep = 0.5;
    segParams.threshold = 0.5;
    segParams.seedClearancePx = seedClearancePx;

    // Получение меток
    LabelsInfo labels = LabelMapping::computeLabels(binaryDilated, /*invert=*/false);

    // Этап сегментации
    auto zones = segmentByGaussianThreshold(binaryDilated, labels, segParams);

    // Создание матрицы зон
    cv::Mat1i segmentation = cv::Mat::zeros(binaryDilated.size(), CV_32S);
    for (const auto &z : zones) {
        segmentation.setTo(z.label, z.mask);
    }

    // Генерация графа связности зон
    ZoneGraph graph;
    buildGraph(graph, zones, segmentation, mapInfo, labels.centroids);

    // Отрисовка графа связности поверх кадрированного/выравненного изображения
    cv::Mat vis = renderZonesOverlay(zones, aligned, cropInfo, 0.65);
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
