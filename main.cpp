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

#define SHOW_DEBUG_IMAGES

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

//    showMat("Raw Map", raw);

    // Заполнение структуры MapInfo
    mapInfo.resolution = resolution;
    mapInfo.originX = map_origin[0];
    mapInfo.originY = map_origin[1];
    mapInfo.theta = map_origin[2];
    mapInfo.height = raw.rows;
    mapInfo.width = raw.cols;

    cv::Mat raw8u;
    raw.convertTo(raw8u, CV_8UC1);

    cv::Mat aligned;
    MapPreprocessing::mapAlign(raw8u, aligned, segmenterConfig.alignmentConfig);

   showMat("aligned Map", aligned);

    // Этап денойзинга.
    auto [rank, cropInfo] = MapPreprocessing::generateDenoisedAlone(aligned, segmenterConfig.denoiseConfig);

   cv::Mat out;
   rank.convertTo(out, CV_8U, 255);
   showMat("Denoised Map", out);

    // Расширенние черных зон(препятствия)
    cv::Mat1b binaryDilated = erodeBinary(rank, segmenterConfig.dilateConfig.kernelSize, segmenterConfig.dilateConfig.iterations);


    cv::Mat1b wallMask;
    cv::compare(binaryDilated, 0, wallMask, cv::CMP_EQ);   // 255 там, где стена
//    cv::Mat1b skeleton;
//    cv::Mat sk8;
//    cv::ximgproc::thinning(wallMask, skeleton, cv::ximgproc::THINNING_ZHANGSUEN);
//    skeleton.convertTo(sk8, CV_8U, 255);
//            std::cout << "skeleton non-zero = " << cv::countNonZero(skeleton) << std::endl;
//    showMat("Skeleton", sk8);

//    std::vector<cv::Point> ends;
//    showMat("Skeleton + endpoints", visualizeSkeletonEndpoints(skeleton, &ends));

//    cv::Mat1b endpointsMask = drawSkeletonEndpoints(skeleton, 3);
//    showMat("Endpoints mask", endpointsMask); // inflation radius

    LabelsInfo labels;
    SegmentationParams segParams;
    segParams.legacyMaxIter = 50;
    segParams.legacySigmaStep = 0.5;
    segParams.legacyThreshold = 0.5;
    segParams.useDownsampleSeeds = true;
    segParams.downsampleConfig.maxIter = 60;
    segParams.downsampleConfig.sigmaStart = 1.0;
    segParams.downsampleConfig.sigmaStep = 0.4;
    segParams.downsampleConfig.threshold = 0.55;

    auto zones = segmentByGaussianThreshold(binaryDilated, labels, segParams);

    cv::Mat1i segmentation = cv::Mat::zeros(binaryDilated.size(), CV_32S);
            int iter = 0;
    for (const auto& z : zones) {

        // cv::Mat1b seg8u;
        // z.mask.convertTo(seg8u, CV_8U, 255);       // CV_8U для applyColorMap
        // showMat("Zone №" + std::to_string(z.label), seg8u);

        segmentation.setTo(z.label, z.mask);
    }

    cv::Mat1i seg = segmentation;                     // ваша сегментация (0 = стены)
    double minVal, maxVal;
    cv::minMaxLoc(seg, &minVal, &maxVal);     // нужен maxVal ≥ 1
    cv::Mat1b gray8;
    seg.convertTo(gray8, CV_8U,
                  maxVal ? 255.0 / maxVal : 1.0,   // α (scale)
                  0);                              // β (shift)

    ZoneGraph graph;
    buildGraph(graph, zones, segmentation, mapInfo, labels.centroids);

    cv::Mat baseGray = aligned.empty() ? raw8u : aligned.clone();
    if (baseGray.channels() > 1)
        cv::cvtColor(baseGray, baseGray, cv::COLOR_BGR2GRAY);
    cv::Mat baseColor;
    cv::cvtColor(baseGray, baseColor, cv::COLOR_GRAY2BGR);
    cv::Rect roi(cropInfo.left,
                 cropInfo.top,
                 rank.cols,
                 rank.rows);
    cv::Mat visRoi = renderZonesOverlay(zones, baseGray(roi), 0.65);
    visRoi.copyTo(baseColor(roi));
    mapping::drawZoneGraphOnMap(graph, baseColor, mapInfo);

    cv::imwrite("segmentation_overlay.png", baseColor);
    if(!isHeadlessMode())
    {
        cv::imshow("segmented", baseColor);
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
