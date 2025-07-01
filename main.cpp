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

using namespace mapping;


/**
 * @param  seg        CV_32S: 0 = стены/фон, ≥1 = код зоны
 * @param  winName    название окна для cv::imshow ("" → не показывать)
 * @param  pngPath    куда сохранить PNG ("" → не сохранять)
 * @param  colormap   см. COLORMAP_* (например, COLORMAP_JET, HOT, VIRIDIS …)
 */
cv::Mat3b colorizeSegmentation(const cv::Mat1i& seg,
                               const cv::Mat1b& wallMask,
                               const std::string& winName  = "Segmentation",
                               std::string      pngPath  = "segmentation.png",
                               int              colormap = cv::COLORMAP_JET)
{
    CV_Assert(!seg.empty() && seg.type() == CV_32S);

    /* 1.  нормируем [0 .. maxLabel] → [0 .. 255]  ------------------------- */
    double minVal, maxVal;
    cv::minMaxLoc(seg, &minVal, &maxVal);
    double scale = (maxVal > 0) ? 255.0 / maxVal : 1.0;

    cv::Mat1b seg8u;
    seg.convertTo(seg8u, CV_8U, scale);       // CV_8U для applyColorMap

    /* 2.  применяем цветовую карту  --------------------------------------- */
    cv::Mat3b color;
//    cv::Mat1i hashed = (seg * 1315423911u) & 0xFF;   // простое хеш-перемешивание
//    hashed.convertTo(seg8u, CV_8U);
    cv::applyColorMap(seg8u, color, colormap);

    /* 3.  делаем фон серым для контраста (опц.) -------------------- */
    // фон = те же пиксели, что были 0 в seg (т.е. seg8u == 0)
    color.setTo(cv::Vec3b(50,50,50), seg8u == 0);

    // добавляем стены
    double alpha = 0.9;                  // непрозрачность 0…1 (0.6 = 60 %)
    cv::Scalar wallColor(0, 0, 0);     // чёрные стены (BGR)

    /* создаём BGR-картинку только для стен */
    cv::Mat overlay(color.size(), color.type(), wallColor);

    /* смешиваем: dst = imgColor*(1-α) + overlay*α  только там, где стена */
    cv::Mat blended = color.clone();
    overlay.copyTo(blended, wallMask);   // overlay → blended (там, где стена)

    /* linearBlend = img*(1-α) + blended*α, но только по маске */
    cv::addWeighted(color, 1.0, blended, alpha, 0.0, blended);


    /* 4.  вывод / сохранение --------------------------------------------- */
    if (!winName.empty()) {
        showMat(winName, blended);
        cv::waitKey(0);
    }
    if (!pngPath.empty())
        cv::imwrite(pngPath, blended);


    return blended;                              // на случай дальнейшей работы
}

/**
 * @brief Построить маску области, достижимой из заданного центроида по свободным
 *        пикселям (значение 0) с помощью Flood Fill.
 *
 * @param freeMap    Одноканальная карта (CV_8UC1), где 0 — свободная ячейка,
 *                   любой ненулевой байт — препятствие/стена.
 * @param seed       Точка-центроид, откуда начинается заливка.
 * @param connectivity 4 или 8 (по умолчанию 8).
 * @return cv::Mat1b того же размера, где 255 помечает залитую область,
 *         0 — остальное.
 *
 * Примечания
 * ----------
 * 1.  Исходная карта **не** изменяется (используется FLOODFILL_MASK_ONLY).
 * 2.  Маска floodFill требует рамку в 1 пиксель; она удаляется перед возвратом.
 * 3.  Если seed лежит на препятствии, функция вернёт нулевую маску.
 */


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

//    showMat("aligned Map", aligned);

    // Этап денойзинга.
    auto [rank, cropInfo] = MapPreprocessing::generateDenoisedAlone(aligned, segmenterConfig.denoiseConfig);

//    showMat("Denoised Map", rank);

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
    auto zones = segmentByGaussianThreshold(binaryDilated, labels, 50, 0.5);

    cv::Mat1i segmentation = cv::Mat::zeros(binaryDilated.size(), CV_32S);
            int iter = 0;
    for (const auto& z : zones) {

        cv::Mat1b seg8u;
        z.mask.convertTo(seg8u, CV_8U, 255);       // CV_8U для applyColorMap
        showMat("Zone №" + std::to_string(z.label), seg8u);

        segmentation.setTo(z.label, z.mask);
    }

    cv::Mat1i  seg = segmentation;                     // ваша сегментация (0 = стены)
    double minVal, maxVal;
    cv::minMaxLoc(seg, &minVal, &maxVal);     // нужен maxVal ≥ 1
    cv::Mat1b gray8;
    seg.convertTo(gray8, CV_8U,
                  maxVal ? 255.0 / maxVal : 1.0,   // α (scale)
                  0);                              // β (shift)

    cv::imshow("Segmentation", gray8);

    ZoneGraph graph;
    buildGraph(graph, zones, segmentation, labels.centroids);

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

    cv::Mat3b vis = colorizeSegmentation(segmentation, wallMask,
                    "Rooms colored",
                    "rooms.png",
                    cv::COLORMAP_JET);



    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);
}
