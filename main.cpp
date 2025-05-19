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
#include "config.hpp"
#include "utils.hpp"

using namespace mapping;

void exampleZoneGraph(){
    ZoneGraph g;
    std::unordered_map<ZoneType, NodePtr> node;   // для удобства соединений

    auto add = [&](ZoneType t, Point2d c)
    {
        double area = 12.0;                           // фиктивные метрики
        double per  = 14.0;
        node[t] = g.addNode(t, area, per, c,
                            {{{c.x-0.5,c.y-0.5},{c.x+0.5,c.y-0.5},
                              {c.x+0.5,c.y+0.5},{c.x-0.5,c.y+0.5}}});
    };

    /* --- размещаем вершины (координаты в «метрах») ---------------------- */
    add(ZoneType::HallVestibule,           { 0,  0});
    add(ZoneType::DoorArea,                { 1,  2});
    add(ZoneType::Corridor,                { 3,  2});
    add(ZoneType::AtriumLobby,             { 6,  0});
    add(ZoneType::Staircase,               { 6, -3});
    add(ZoneType::ElevatorZone,            { 8, -3});
    add(ZoneType::LivingRoomOfficeBedroom, { 0,  5});
    add(ZoneType::Kitchenette,             { 3,  5});
    add(ZoneType::Sanitary,                { 6,  5});
    add(ZoneType::NarrowConnector,         { 5,  2});
    add(ZoneType::StorageUtility,          { 7,  2});
    add(ZoneType::Unknown,                 {-2,  2});

    /* --- рёбра с реалистичными ширинами --------------------------------- */
    auto W = [](double m){ return m; };        // удобный alias

    g.connectZones(node[ZoneType::HallVestibule]->id(),
                   node[ZoneType::DoorArea]->id(), W(1.4));

    g.connectZones(node[ZoneType::DoorArea]->id(),
                   node[ZoneType::Corridor]->id(), W(1.2));

    g.connectZones(node[ZoneType::AtriumLobby]->id(),
                   node[ZoneType::Corridor]->id(), W(2.0));

    g.connectZones(node[ZoneType::AtriumLobby]->id(),
                   node[ZoneType::Staircase]->id(), W(1.6));

    g.connectZones(node[ZoneType::AtriumLobby]->id(),
                   node[ZoneType::ElevatorZone]->id(), W(1.6));

    /* комнаты выходят в коридор через двери */
    g.connectZones(node[ZoneType::Corridor]->id(),
                   node[ZoneType::LivingRoomOfficeBedroom]->id(), W(1.0));

    g.connectZones(node[ZoneType::Corridor]->id(),
                   node[ZoneType::Kitchenette]->id(), W(0.9));

    g.connectZones(node[ZoneType::Corridor]->id(),
                   node[ZoneType::Sanitary]->id(), W(0.8));

    /* узкий проход к кладовой */
    g.connectZones(node[ZoneType::Corridor]->id(),
                   node[ZoneType::NarrowConnector]->id(), W(0.6));

    g.connectZones(node[ZoneType::NarrowConnector]->id(),
                   node[ZoneType::StorageUtility]->id(), W(0.8));

    /* неизвестная зона примыкает к коридору */
    g.connectZones(node[ZoneType::Unknown]->id(),
                   node[ZoneType::Corridor]->id(), W(1.0));

    /* ---------- DOT‑файл ----------------------------------------------- */
    std::ofstream dot("graph.dot");
    writeDot(g, dot);

    /* ---------- Быстрый предварительный просмотр (OpenCV) -------------- */
    cv::Mat img;
    drawZoneGraph(g, img, 50.0 /*px/м*/, 7 /*радиус*/, true /*подписи ширины*/);
    cv::imshow("Floor‑Plan Graph", img);
    cv::imwrite("graph_preview.png", img);
}

/**
 * @param  seg        CV_32S: 0 = стены/фон, ≥1 = код зоны
 * @param  winName    название окна для cv::imshow ("" → не показывать)
 * @param  pngPath    куда сохранить PNG ("" → не сохранять)
 * @param  colormap   см. COLORMAP_* (например, COLORMAP_JET, HOT, VIRIDIS …)
 */
cv::Mat3b colorizeSegmentation(const cv::Mat1i& seg,
                               std::string      winName  = "Segmentation",
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
    cv::applyColorMap(seg8u, color, colormap);

    /* 3.  делаем стены/фон серыми для контраста (опц.) -------------------- */
    // стены = те же пиксели, что были 0 в seg (т.е. seg8u == 0)
    color.setTo(cv::Vec3b(50,50,50), seg8u == 0);

    /* 4.  вывод / сохранение --------------------------------------------- */
    if (!winName.empty()) {
        cv::imshow(winName, color);
        cv::waitKey(0);
    }
    if (!pngPath.empty())
        cv::imwrite(pngPath, color);

    return color;                              // на случай дальнейшей работы
}

std::vector<ZoneMask>
segmentByWallErosion(cv::Mat1b map,            // чёрный = стена, белый = пусто
                     int maxIter = 100,
                     int erosionSize = 1)
{
    /* 1. рассчитываем все центроиды свободных областей -------------------- */
    auto info = LabelMapping::computeLabels(map, 2*erosionSize+1);     // фон=0 уже стена

    cv::Mat1b free = map;                    // белое = свободно
    cv::Mat1b kernel = cv::getStructuringElement(
                           cv::MORPH_RECT, cv::Size(2*erosionSize+1, 2*erosionSize+1));

    std::vector<ZoneMask> allZones;          // общий результат

    /* 2. делаем копию centroids, чтоб удалять «уже изолированные» ---------- */
    std::unordered_map<int, cv::Point> todo = info.centroids;

    for (int iter = 0; iter < maxIter && !todo.empty(); ++iter)
    {
        cv::Mat1b seg8u;
        map.convertTo(seg8u, CV_8U, 255);       // CV_8U для applyColorMap
        showMat("Map " + std::to_string(iter), seg8u);
        /* 2.1  берём **текущие** ещё не изолированные центроиды */
        auto zones = LabelMapping::extractIsolatedZones(free, todo, /*invertFree=*/true);

        /* 2.2  сохраняем найденные зоны и удаляем их из todo */
        for (auto& z : zones) {
            allZones.push_back( std::move(z) );
            todo.erase(allZones.back().label);          // убрать метку из «работы»
        }

        if (todo.empty()) break;                        // всё изолировали

        /* 2.3  «толстим» стены чёрным -> erode (шаг = erosionSize) */
        cv::erode(map, map, kernel, cv::Point(-1,-1), 1);   // стены растут
        free = map;                                         // свободно = белое
    }

    if (!todo.empty())
        std::cerr << "[warn] maxIter reached, "
                  << todo.size() << " zones still not isolated\n";

    return allZones;                                        // набор масок зон
}

std::vector<ZoneMask>
segmentByGaussianThreshold(const cv::Mat1b& srcBinary,   // 0 = стены, 255 = пусто
                           int   maxIter      = 40,
                           double sigmaStep   = 1.0,     // приращение σ на итерацию
                           int    threshold   = 0.5)     // T для THRESH_BINARY
{
    CV_Assert(!srcBinary.empty() && srcBinary.type() == CV_8UC1);

    /* 0.  центроиды свободных областей исходной карты --------------------- */
    auto info = LabelMapping::computeLabels(srcBinary, /*invert=*/false);

    std::unordered_map<int, cv::Point> todo = info.centroids; // ещё не изолированы
    std::vector<ZoneMask> allZones;                           // результат

    cv::Mat blurred;
    cv::Mat bin;

    for (int iter = 0; iter < maxIter && !todo.empty(); ++iter)
    {
        double sigma = (iter + 1) * sigmaStep;               // растёт σ

        /* 0.  подготовка маски стен с усиленной маской в endpoints --------------------------- */
        // Расширяем стены
        cv::Mat1b kernel = cv::getStructuringElement(
                               cv::MORPH_RECT, cv::Size(2*1+1, 2*1+1));
        cv::erode(srcBinary, srcBinary, kernel, cv::Point(-1,-1), 1);

        /* 1.  размытие исходной бинарной карты --------------------------- */
        cv::GaussianBlur(srcBinary, blurred, cv::Size(0,0), sigma, sigma,
                         cv::BORDER_REPLICATE);

        /* 2.  порог → новая бинарная карта «виртуально утолщённых» стен -- */
        cv::threshold(blurred, bin, threshold, 1, cv::THRESH_BINARY);

        cv::Mat1b seg8u;
        bin.convertTo(seg8u, CV_8U, 255);       // CV_8U для applyColorMap
//        showMat("srcBinary " + std::to_string(iter), seg8u);

        if (cv::countNonZero(bin) == 0) {        // всё стало стенами
            std::cerr << "[warn] free space vanished at iter " << iter << '\n';
            break;
        }

        /* 3.  ищем компоненты, изолированы ли центроиды ------------------- */
        auto zones = LabelMapping::extractIsolatedZones(bin, todo, /*invertFree=*/true);

        int it = 0;
        for (auto& z : zones) {
            cv::dilate(z.mask, z.mask, kernel, cv::Point(-1,-1), iter);
            allZones.push_back( std::move(z) );
            todo.erase(z.label);   // метка выполнена
        }
    }

    if (!todo.empty())
        std::cerr << "[warn] maxIter reached, "
                  << todo.size() << " zones still not isolated\n";

    return allZones;                             // набор масок зон
}


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

    showMat("Raw Map", raw);

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

    showMat("Denoised Map", rank);

    // Расширенние черных зон(препятствия)
    cv::Mat1b binaryDilated = erodeBinary(rank, segmenterConfig.dilateConfig.kernelSize, segmenterConfig.dilateConfig.iterations);







            cv::Mat1b wallMask;
            cv::compare(binaryDilated, 0, wallMask, cv::CMP_EQ);   // 255 там, где стена
    cv::Mat1b skeleton;
    cv::Mat sk8;
    cv::ximgproc::thinning(wallMask, skeleton, cv::ximgproc::THINNING_ZHANGSUEN);
    skeleton.convertTo(sk8, CV_8U, 255);
            std::cout << "skeleton non-zero = " << cv::countNonZero(skeleton) << std::endl;
    showMat("Skeleton", sk8);

    std::vector<cv::Point> ends;
    showMat("Skeleton + endpoints", visualizeSkeletonEndpoints(skeleton, &ends));

    cv::Mat1b endpointsMask = drawSkeletonEndpoints(skeleton, 3);
    showMat("Endpoints mask", endpointsMask); // inflation radius





    auto zones = segmentByGaussianThreshold(binaryDilated, 50, 0.5);

    cv::Mat1i segmentation = cv::Mat::zeros(binaryDilated.size(), CV_32S);
            int iter = 0;
    for (const auto& z : zones) {

        cv::Mat1b seg8u;
        z.mask.convertTo(seg8u, CV_8U, 255);       // CV_8U для applyColorMap
//        showMat("Zone №" + std::to_string(iter++), seg8u);

        segmentation.setTo(z.label, z.mask);
    }

    cv::Mat3b vis = colorizeSegmentation(segmentation,
                    "Rooms colored",
                    "rooms.png",
                    cv::COLORMAP_VIRIDIS);

    exampleZoneGraph();

    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);
}
