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

/**
 * @brief High level segmentation pipeline orchestrating all steps.
 */
class Segmenter {
public:
    /**
     * Run the full segmentation pipeline on the given image.
     * @param raw     Source image in grayscale or float32 [0,1].
     * @param config  Segmentation parameters.
     * @return        Final segmentation map.
     */
    cv::Mat doSegment(const cv::Mat& raw, const SegmenterConfig& config);
private:
    /** Generate denoised map and cropping information. */
    std::pair<cv::Mat, Segmentation::CropInfo> generateDenoisedAlone(const cv::Mat& raw,
                                                                      const DenoiseConfig& config);
    /** Compute list of labels for different sigma values. */
    std::tuple<std::vector<double>, std::vector<cv::Mat>, std::vector<int>, std::vector<cv::Mat>>
    computeLabelsList(const cv::Mat& binaryDilated,
                      double sigmaStart, double sigmaStep, int maxIter,
                      int backgroundErosionKernelSize, double gaussianSeedsThreshold, bool debug);
    /** Over-segmentation stage producing global label map. */
    std::pair<cv::Mat, int> overSegment(const cv::Mat& ridges,
                                        const std::vector<double>& sigmasList,
                                        const std::vector<cv::Mat>& labelsList,
                                        const OverSegmentConfig& config);
    /** Prepare segments for export: uncrop and remap borders. */
    cv::Mat prepareSegmentsForExport(const cv::Mat& seg, const Segmentation::CropInfo& cropInfo);

    /** Find local maxima of the distance transform. */
    static cv::Mat getLocalMax(const cv::Mat& dist);
    static cv::Mat getCentroids(const cv::Mat& binaryImage);
    static cv::Mat assignLabels(const cv::Mat& binaryImage);
    static cv::Mat getSeeds(const cv::Mat& binInput,
                     int dilateIterations = 1,
                     int dilateKernelSize = 3);
    static cv::Mat performWatershed(const cv::Mat& dist, const cv::Mat& labels);
};

#endif // SEGMENTER_H
