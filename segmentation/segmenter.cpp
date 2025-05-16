#include "segmenter.hpp"
#include <iostream>

// Реализация генерации денойзенной карты.
// Здесь вызываются функции из Segmentation для бинаризации, кадрирования, инверсии и удаления шумовых компонентов.
std::pair<cv::Mat, Segmentation::CropInfo> Segmenter::generateDenoisedAlone(const cv::Mat& raw,
    const DenoiseConfig& config) {

    // Создаем бинарную карту для определения области кадрирования.
    cv::Mat binaryForCrop = makeBinary(raw, config.binaryForCropThreshold * 255, 255);
    Segmentation::CropInfo cropInfo = Segmentation::cropSingleInfo(binaryForCrop, config.cropPadding);

    // Освобождаем память, если необходимо.
    // Обрезаем исходное изображение до ROI.
    cv::Mat rank = Segmentation::cropBackground(raw, cropInfo);

    // Применяем бинаризацию для удаления шума.
    rank = makeBinary(rank, config.binaryThreshold * 255, 255);

    // Инвертируем изображение.
    rank = makeInvert(rank);
    // Удаляем маленькие компоненты (снаружи).
    rank = Segmentation::removeSmallConnectedComponents(rank, "fixed", config.compOutMinSize, 4, false);
    // Инвертируем снова.
    rank = makeInvert(rank);
    // Удаляем маленькие компоненты (внутри).
    rank = Segmentation::removeSmallConnectedComponents(rank, "fixed", config.compInMinSize, 4, false);

    // Финальная бинаризация.
    rank = makeBinary(rank, config.rankBinaryThreshold);
    return { rank, cropInfo };
}

// Функция выполняет watershed.
// dist – исходное изображение (например, distance transform).
// labels – матрица маркеров, где каждая область имеет уникальное целое значение, фон – 0.
// Функция возвращает результирующую матрицу типа CV_32SC1 с разметкой watershed.
cv::Mat Segmenter::performWatershed(const cv::Mat& dist, const cv::Mat& labels)
{
    // Преобразуем исходное изображение в формат CV_8UC3, так как watershed требует цветное изображение.
    cv::Mat src;
    if (dist.type() == CV_8UC1) {
        cv::cvtColor(dist, src, cv::COLOR_GRAY2BGR);
    } else if (dist.type() == CV_32FC1) {
        // Нормализуем и конвертируем 32F в 8U.
        cv::Mat distNormalized, dist8U;
        cv::normalize(dist, distNormalized, 0, 255, cv::NORM_MINMAX);
        distNormalized.convertTo(dist8U, CV_8U);
        cv::cvtColor(dist8U, src, cv::COLOR_GRAY2BGR);
    } else if (dist.type() == CV_8UC3) {
        src = dist.clone();
    } else {
        // Если тип отличается, попробуем привести к 8UC1 и затем в 8UC3.
        cv::Mat temp;
        dist.convertTo(temp, CV_8U);
        cv::cvtColor(temp, src, cv::COLOR_GRAY2BGR);
    }

    // Подготавливаем маркеры: watershed требует матрицу типа CV_32SC1.
    cv::Mat markers;
    if (labels.type() != CV_32SC1) {
        labels.convertTo(markers, CV_32S);
    } else {
        markers = labels.clone();
    }

    // Вызываем watershed. В результате, границы будут обозначены как -1.
    try {
        cv::watershed(src, markers);
    }
    catch (const cv::Exception& e) {
        std::cerr << "cv::watershed exception: " << e.what() << std::endl;
        throw;
    }

    return markers;
}

void showWatershedResult(const cv::Mat& markers)
{
    // Создаем изображение для визуализации
    cv::Mat result(markers.size(), CV_8UC3, cv::Scalar(0,0,0));

    // Определим максимальное значение меток (исключая -1)
    int maxLabel = 0;
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int label = markers.at<int>(i, j);
            if (label > maxLabel)
                maxLabel = label;
        }
    }

    // Генерируем случайные цвета для каждого сегмента
    std::vector<cv::Vec3b> colors(maxLabel + 1);
    colors[0] = cv::Vec3b(0, 0, 0); // фон, если требуется
    for (int i = 1; i <= maxLabel; i++) {
        colors[i] = cv::Vec3b(std::rand() % 256, std::rand() % 256, std::rand() % 256);
    }

    // Заполняем изображение результатами
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index == -1) {
                // Границы. Можно выбрать любой цвет, например, белый или синий.
                result.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
            }
            else if (index <= 0 || index > maxLabel) {
                // Фон или некорректные метки
                result.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
            }
            else {
                result.at<cv::Vec3b>(i, j) = colors[index];
            }
        }
    }

    // Вывод результата через imshow
    cv::imshow("6 - Watershed Result", result);
    cv::waitKey(0);
}


// Примерная реализация overSegment.
// Здесь производится remapping локальных меток, вычисление глобальных мапперов,
// генерация карты начальных семян и финальная watershed-сегментация.
std::pair<cv::Mat, int> Segmenter::overSegment(const cv::Mat& ridges,
    const std::vector<double>& sigmasList,
    const std::vector<cv::Mat>& labelsList,
    const OverSegmentConfig& config) {

    auto mappers = LabelMapping::findConsistentGlobalLabels(sigmasList, labelsList);
    auto lifespan = LabelMapping::findLifespans(mappers);
    auto mapInitialSeeds = LabelMapping::generateMapInitialSeeds(lifespan, mappers, labelsList, "start", 0, {});
    cv::Mat superposedLabels = LabelMapping::superposeMapInitialSeeds(mapInitialSeeds, labelsList);

    // Финальная watershed-сегментация.
    cv::Mat overSegmented = Segmentation::createWatershedSegment(ridges, superposedLabels, 4);
    double dummyMin, borderMax;
    cv::minMaxLoc(overSegmented, &dummyMin, &borderMax);
    int borderLabel = static_cast<int>(borderMax);
    return { overSegmented, borderLabel };
}

// Применяет uncrop_background и remap_border для подготовки карты к экспорту.
cv::Mat Segmenter::prepareSegmentsForExport(const cv::Mat& seg, const Segmentation::CropInfo& cropInfo) {
    cv::Mat segUncropped = Segmentation::uncropBackground(seg, cropInfo, cv::Scalar(0));
    cv::Mat segRemapped = Segmentation::remapBorder(segUncropped);
    return segRemapped;
}

// Основной метод сегментации, объединяющий все этапы.
cv::Mat Segmenter::doSegment(const cv::Mat& raw, const SegmenterConfig& config) {
    cv::Mat rawAligned = raw;
    double alignmentAngle = 0.0;
    if (config.alignmentConfig.enable) {
        // Находим угол выравнивания с помощью MapAlignment.
        alignmentAngle = MapPreprocessing::findAlignmentAngle(raw, config.alignmentConfig);
        rawAligned = MapPreprocessing::rotateImage(raw, alignmentAngle);
    }

    std::cout << "Type raw: " << type2str(raw.type()) << std::endl;
    // Этап денойзинга.
    auto [rank, cropInfo] = generateDenoisedAlone(rawAligned, config.denoiseConfig);

    // Расширенние черных зон(препятствия)
    cv::Mat binaryDilated = erodeBinary(rank, config.dilateConfig.kernelSize, config.dilateConfig.iterations);
     cv::Mat binaryDilatedOut;
    binaryDilated.convertTo(binaryDilatedOut, CV_8U, 255);
    cv::imshow("2 - binaryDilated", binaryDilatedOut);




    // Вычисляем список меток.
    auto [sigmasList, labelsList, nccsList, segmentsList] = computeLabelsList(binaryDilated,
        config.labelsListConfig.sigmaStart, config.labelsListConfig.sigmaStep,
        config.labelsListConfig.maxIter, config.labelsListConfig.backgroundErosionKernelSize,
        config.labelsListConfig.gaussianSeedsThreshold, config.labelsListConfig.debug);

    // Если sigmaStep == 0, используем первую сегментированную карту.
    if (config.labelsListConfig.sigmaStep == 0) {
        return prepareSegmentsForExport(segmentsList[0], cropInfo);
    }




    // Этап over-segmentation.
    auto [segments, borderLabel] = overSegment(binaryDilated, sigmasList, labelsList, config.overSegmentConfig);

    cv::imshow("7 - segments", segments);
    // Построение графа на основе сегментов.
    MapGraph mapGraph(segments, borderLabel, 0.04);
    auto [G, commonBordersMap, borders] = mapGraph.findGraphRepresentation();

    // Слияние узлов по критерию маленькой площади.
    GraphMerger::mergeNodesInPlace(G, segments, commonBordersMap, borders, borderLabel,
                                   { config.mergeNodesConfig.areaThreshold });
//    debugger::add("G2", static_cast<int>(G.nodes.size()));
//    debugger::add("segments_merged_by_node", segments);

    // Слияние рёбер по критерию длин.
    GraphMerger::mergeEdgesInPlace(G, segments, commonBordersMap, borders, borderLabel,
                                   { config.mergeEdgesConfig.lengthThreshold });
//    debugger::add("G3", static_cast<int>(G.nodes.size()));
//    debugger::add("segments_merged_by_edge", segments);

    // Выводим информацию по зонам.
    mapGraph.printZonesInfo();
    std::cout << mapGraph.generatePDDLProblem("problemName", "domainName") << std::endl;

    // Подготовка финальной карты для экспорта.
    cv::Mat finalMap = prepareSegmentsForExport(segments, cropInfo);
    return finalMap;
}
