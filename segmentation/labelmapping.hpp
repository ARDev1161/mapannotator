#ifndef LABELMAPPING_H
#define LABELMAPPING_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <string>
#include <unordered_map>
#include <utility> // для std::pair
#include "../utils.hpp"

/** Information about detected label centroids. */
struct LabelsInfo {
    cv::Mat1i                          centroidLabels;   ///< 0 = background, >0 = label id at centroid
    std::unordered_map<int, cv::Point> centroids;        ///< label → (x,y)
    int                                numLabels;        ///< number of labels including background
};

/** Binary mask of a single labelled zone. */
struct ZoneMask {
    int      label;   ///< zone id (≥1)
    cv::Mat1b mask;   ///< 255 inside, 0 outside (same size as free map)
};

/**
 * @brief Helper algorithms dealing with label maps.
 */
class LabelMapping {
public:
    /** Deep-copy the map and replace pixels using remapDict (expects CV_32S). */
    template<typename T>
    static cv::Mat remap(const cv::Mat& inputMap, const std::map<T, T>& remapDict);

    /** Compute label centroids and IDs from a dilated binary map. */
    static LabelsInfo computeLabels(const cv::Mat1b& binaryDilated, int backgroundErosionKernelSize);

    static std::vector<ZoneMask>
    extractIsolatedZones(const cv::Mat1b& freeMap,
                         const std::unordered_map<int, cv::Point>& centroids,
                         bool invertFree = false);
    /**
     *  buildOccupancyMask
     *  ------------------
     *  Формирует итоговую карту занятости:
     *    • 0  – занято  (стены **или** зоны)
     *    • 255 – свободно
     *
     *  @param background  CV_8UC1, 0 = стена, 255 = пусто
     *  @param allZones    вектор зон (mask: 255 = зона, 0 = остальное)
     *  @return            CV_8UC1 той же геометрии, что background
     */
    static cv::Mat1b buildOccupancyMask(const cv::Mat1b&                   background,
                                 const std::vector<ZoneMask>&       allZones);
private:
    // Функция получает бинарное изображение (8UC1), где объекты имеют значение 255, фон — 0,
    // и возвращает карту меток для watershed (каждая связная область получает уникальное целочисленное значение).
    /** Helper for local maxima extraction used in watershed seeding. */
    static cv::Mat getLocalMax(const cv::Mat& dist);

    /**
     *  @brief  Формирует Sparse‑карту центроидов: в каждой такой точке
     *          значение пиксела равно коду её метки (CV_32S).
     *
     *  @param  binary   входная маска 8 бит; 0 = фон, ≠0 = объект
     *  @param  invert   true, если объекты чёрные (нужно инвертировать)
     */
    static LabelsInfo getCentroids(const cv::Mat1b& binaryImage, bool invert = false);

    static LabelsInfo getSeeds(const cv::Mat& binInput,
                                 int dilateIterations = 1,
                                 int dilateKernelSize = 3);
    /**
     * @brief  Детектировать углы Harris и добавить окружности в тот же wallMask.
     *
     * @param wallMask   [in/out] Маска стен (CV_8UC1): 0 — стены, 255 — свободно.
     *                   Функция модифицирует эту матрицу.
     * @param radius     Радиус окружности.
     * @param circleVal  Значение пикселей окружности (0 = стена, 255 = свободно, др. = метка).
     * @param maxCorners Максимальное число углов (по умолчанию 400).
     * @param harrisK    Параметр k для Harris (0.04–0.06).
     * @param quality    Относительный порог (0‒1) для goodFeaturesToTrack.
     * @param minDist    Мин. расстояние между углами (обычно ≥ radius).
     *
     * @return           Количество нарисованных окружностей.
     */
    static int addCornerCirclesHarris(cv::Mat1b &wallMask,
                                   int radius,
                                   uchar circleVal  = 0,
                                   int maxCorners  = 400,
                                   double harrisK   = 0.04,
                                   double quality  = 0.01,
                                   double minDist  = 5.0);
};

// remap: глубокое копирование и замена значений согласно словарю remapDict.
template<typename T>
cv::Mat LabelMapping::remap(const cv::Mat& inputMap, const std::map<T, T>& remapDict) {
    // Проверяем, что тип матрицы соответствует ожидаемому.
    CV_Assert(inputMap.depth() == cv::DataType<T>::depth);

    // Клонируем входную матрицу, чтобы не изменять исходные данные.
    cv::Mat mapNew = inputMap.clone();
    for (int i = 0; i < mapNew.rows; i++) {
        for (int j = 0; j < mapNew.cols; j++) {
            int val = mapNew.at<int>(i, j);
            auto it = remapDict.find(val);
            if (it != remapDict.end()) {
                mapNew.at<int>(i, j) = it->second;
            }
        }
    }
    return mapNew;
}

#endif // LABELMAPPING_H
