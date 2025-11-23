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
    /**
     * @brief Generate denoised map and cropping information.
     *
     * @param raw     Input grayscale map.
     * @param config  Denoising parameters.
     * @return        Binary denoised map and crop offsets.
     */
    std::pair<cv::Mat, Segmentation::CropInfo> generateDenoisedAlone(const cv::Mat& raw,
                                                                      const DenoiseConfig& config);
    /**
     * @brief Compute list of labels for different sigma values.
     *
     * @param binaryDilated  Preprocessed binary map.
     * @param sigmaStart     Initial Gaussian sigma.
     * @param sigmaStep      Sigma increment.
     * @param maxIter        Maximum iterations.
     * @param backgroundErosionKernelSize  Kernel for background erosion.
     * @param gaussianSeedsThreshold       Threshold for seed generation.
     * @param debug          Emit intermediate images when true.
     * @return               Tuple of sigmas, label maps, background labels, and seed masks.
     */
    std::tuple<std::vector<double>, std::vector<cv::Mat>, std::vector<int>, std::vector<cv::Mat>>
    computeLabelsList(const cv::Mat& binaryDilated,
                      double sigmaStart, double sigmaStep, int maxIter,
                      int backgroundErosionKernelSize, double gaussianSeedsThreshold, bool debug);
    /**
     * @brief Over-segmentation stage producing global label map.
     *
     * @param ridges      Distance-transform ridges.
     * @param sigmasList  Sigma values used.
     * @param labelsList  Candidate labels per sigma.
     * @param config      Over-segmentation configuration.
     * @return            Pair of merged label map and number of labels.
     */
    std::pair<cv::Mat, int> overSegment(const cv::Mat& ridges,
                                        const std::vector<double>& sigmasList,
                                        const std::vector<cv::Mat>& labelsList,
                                        const OverSegmentConfig& config);
    /**
     * @brief Prepare segments for export: uncrop and remap borders.
     *
     * @param seg       Cropped segmentation map.
     * @param cropInfo  Crop offsets to restore.
     * @return          Uncropped segmentation map with border remapped.
     */
    cv::Mat prepareSegmentsForExport(const cv::Mat& seg, const Segmentation::CropInfo& cropInfo);

    /** Find local maxima of the distance transform. */
    static cv::Mat getLocalMax(const cv::Mat& dist);
    /** Compute centroids of connected components. */
    static cv::Mat getCentroids(const cv::Mat& binaryImage);
    /** Assign unique labels to connected components. */
    static cv::Mat assignLabels(const cv::Mat& binaryImage);
    /**
     * @brief Build seed markers by dilating free space.
     * @param binInput          Binary input map.
     * @param dilateIterations  Number of dilations.
     * @param dilateKernelSize  Kernel size for dilation.
     */
    static cv::Mat getSeeds(const cv::Mat& binInput,
                     int dilateIterations = 1,
                     int dilateKernelSize = 3);
    /**
     * @brief Run watershed on distance transform and seeds.
     * @param dist    Distance transform image.
     * @param labels  Initial markers.
     */
    static cv::Mat performWatershed(const cv::Mat& dist, const cv::Mat& labels);
};

#endif // SEGMENTER_H
