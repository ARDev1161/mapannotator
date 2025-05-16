#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <filesystem>
#include <set>
#include "../utils.hpp"

class Segmentation {
public:
    // Структура для хранения информации о кадрировании.
    struct CropInfo {
        int left;   // Отступ слева
        int top;    // Отступ сверху
        int right;  // Отступ справа
        int bottom; // Отступ снизу
    };

    // Находит ограничивающий прямоугольник (bounding box) для карты.
    // Алгоритм использует значение фонового пикселя (пиксель [0,0]).
    // Возвращается cv::Rect, где (x,y) – верхний левый угол, а width и height – размеры ROI.
    static cv::Rect findBB(const cv::Mat& map);

    // Вычисляет информацию для единичного кадрирования (с отступами).
    static CropInfo cropSingleInfo(const cv::Mat& map, int padding = 0);
    // Вычисляет информацию для пакетного кадрирования двух карт.
    static CropInfo cropBundleInfo(const cv::Mat& rankMap, const cv::Mat& trackMap);

    // Функции кадрирования и восстановления исходного размера.
    static cv::Mat cropBackground(const cv::Mat& map, const CropInfo& info);
    static cv::Mat uncropBackground(const cv::Mat& map, const CropInfo& info, const cv::Scalar& background);

    // Функция remapBorder: заменяет пиксели с заданным ярлыком границы на 255.
    // Если border_label == -1, то используется максимальное значение.
    static cv::Mat remapBorder(const cv::Mat& map, int border_label = -1);

    // Удаляет небольшие связные компоненты из бинарного изображения.
    // Параметр method может принимать значения "fixed", "mean" или "median".
    static cv::Mat removeSmallConnectedComponents(const cv::Mat& src, const std::string& method = "fixed",
                                                    int min_size = 1, int connectivity = 4, bool debug = false);

    // Генерирует метки (seeds) на основе гауссового размытия.
    static cv::Mat generateGaussianSeeds(const cv::Mat& src, const cv::Mat& background_mask,
                                           double sigma, double threshold = 0.05);

    // Выполняет сегментацию методом watershed.
    // Входное изображение src должно быть 3-канальным; labels – исходные метки (CV_32S).
    static cv::Mat createWatershedSegment(const cv::Mat& src, const cv::Mat& labels, int connectivity = 4);
};

#endif // SEGMENTATION_H
