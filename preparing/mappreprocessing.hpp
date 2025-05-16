#ifndef MAPPREPROCESSING_H
#define MAPPREPROCESSING_H

#include <opencv2/core.hpp>
#include <vector>
#include "../config.hpp"
#include "../segmentation/segmentation.hpp"

// Класс MapPreprocessing инкапсулирует функциональность выравнивания и денойзинга карты.
class MapPreprocessing {
public:
    // Находит корректировку ориентации карты по границам:
    // возвращает минимальный поворотный прямоугольник и устанавливает angle – угол поворота.
    static cv::RotatedRect findMapOrientationAdjustment(const cv::Mat& im, double& angle);

    // Поворачивает изображение на заданный угол.
    static cv::Mat rotateImage(const cv::Mat& image, double angle);

    // Отрисовывает ограничивающий прямоугольник (bounding box) на изображении.
    static cv::Mat drawBB(const cv::Mat& im, const cv::RotatedRect& rect, const cv::Scalar& color = cv::Scalar(0, 0, 255));

    // Рисует линию на изображении.
    static void drawLine(cv::Mat& im, int x0, int y0, int x1, int y1,
                         const cv::Scalar& color = cv::Scalar(255, 0, 0), int lineThickness = 1);

    // Вычисляет взвешенную гистограмму углов по изображению.
    // rank – обработанное (например, бинарное) изображение (CV_32F или CV_8U).
    static std::vector<double> computeWeightedAngleHistogram(const cv::Mat& rank,
                                                               const AlignmentConfig& alignmentConfig,
                                                               bool debug = false);

    // По гистограмме находит лучший угол выравнивания.
    static double findBestAlignmentAngleFromHistogram(const std::vector<double>& hist,
                                                      const HistogramConfig& histConfig);

    // Полный пайплайн: вычисляет угол выравнивания по входному градационному (grayscale) изображению.
    // Предварительная обработка (бинаризация, crop и т.д.) в данном примере выполняется простым пороговым преобразованием.
    static double findAlignmentAngle(const cv::Mat& grayscale, const AlignmentConfig& config);

    // Реализация генерации денойзенной карты.
    // Здесь вызываются функции из Segmentation для бинаризации, кадрирования, инверсии и удаления шумовых компонентов.
    static std::pair<cv::Mat, Segmentation::CropInfo> generateDenoisedAlone(const cv::Mat& raw, const DenoiseConfig& config);

    static bool mapAlign(const cv::Mat& raw, cv::Mat& out, const AlignmentConfig& config);
};

#endif // MAPPREPROCESSING_H
