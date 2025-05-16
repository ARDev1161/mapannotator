#ifndef SEGMENTER_H
#define SEGMENTER_H

#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>
#include "../preparing/mappreprocessing.hpp"
#include "segmentation.hpp"
#include "labelmapping.hpp"
#include "../utils.hpp"
#include "../config.hpp"

class Segmenter {
public:
    // Основной метод сегментации:
    // Принимает исходное изображение (грейскейл или float32 с диапазоном [0,1]) и конфигурацию.
    // Возвращает глобальную сегментационную карту после всех этапов.
    cv::Mat doSegment(const cv::Mat& raw, const SegmenterConfig& config);
private:
    // Генерация денойзенной карты и информации о кадрировании.
    std::pair<cv::Mat, Segmentation::CropInfo> generateDenoisedAlone(const cv::Mat& raw,
                                                                      const DenoiseConfig& config);
    // Вычисление списка меток для различных sigma.
    std::tuple<std::vector<double>, std::vector<cv::Mat>, std::vector<int>, std::vector<cv::Mat>>
    computeLabelsList(const cv::Mat& binaryDilated,
                      double sigmaStart, double sigmaStep, int maxIter,
                      int backgroundErosionKernelSize, double gaussianSeedsThreshold, bool debug);
    // Этап over-segmentation: создание глобальной карты меток на основе локальных мапперов.
    std::pair<cv::Mat, int> overSegment(const cv::Mat& ridges,
                                        const std::vector<double>& sigmasList,
                                        const std::vector<cv::Mat>& labelsList,
                                        const OverSegmentConfig& config);
    // Подготовка сегментов для экспорта: откадрирование и remap границ.
    cv::Mat prepareSegmentsForExport(const cv::Mat& seg, const Segmentation::CropInfo& cropInfo);

    static cv::Mat getLocalMax(const cv::Mat& dist);
    static cv::Mat getCentroids(const cv::Mat& binaryImage);
    static cv::Mat assignLabels(const cv::Mat& binaryImage);
    static cv::Mat getSeeds(const cv::Mat& binInput,
                     int dilateIterations = 1,
                     int dilateKernelSize = 3);
    static cv::Mat performWatershed(const cv::Mat& dist, const cv::Mat& labels);
};

#endif // SEGMENTER_H
