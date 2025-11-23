#ifndef MAPPREPROCESSING_H
#define MAPPREPROCESSING_H

#include <opencv2/core.hpp>
#include <vector>
#include "../config.hpp"
#include "../segmentation/segmentation.hpp"

/**
 * @brief Utility routines for map alignment and denoising.
 */
class MapPreprocessing {
public:
    /** Find orientation adjustment by analysing map boundaries. */
    static cv::RotatedRect findMapOrientationAdjustment(const cv::Mat& im, double& angle);

    /** Rotate image by a given angle. */
    static cv::Mat rotateImage(const cv::Mat& image, double angle);

    /** Draw bounding box on the image. */
    static cv::Mat drawBB(const cv::Mat& im, const cv::RotatedRect& rect, const cv::Scalar& color = cv::Scalar(0, 0, 255));

    /** Draw a line on the image. */
    static void drawLine(cv::Mat& im, int x0, int y0, int x1, int y1,
                         const cv::Scalar& color = cv::Scalar(255, 0, 0), int lineThickness = 1);

    /** Compute weighted angle histogram for the given image. */
    static std::vector<double> computeWeightedAngleHistogram(const cv::Mat& rank,
                                                               const AlignmentConfig& alignmentConfig,
                                                               bool debug = false);

    /** Find best alignment angle from histogram using moving average. */
    static double findBestAlignmentAngleFromHistogram(const std::vector<double>& hist,
                                                      const HistogramConfig& histConfig);

    /** Full pipeline that computes the alignment angle of a grayscale image. */
    static double findAlignmentAngle(const cv::Mat& grayscale, const AlignmentConfig& config);

    /** Generate a denoised map and cropping info using Segmentation helpers. */
    static std::pair<cv::Mat, Segmentation::CropInfo> generateDenoisedAlone(const cv::Mat& raw, const DenoiseConfig& config);

    /**
     * Resolve unknown (grey) regions by iteratively expanding black and
     * white areas until convergence.
     */
    static cv::Mat unknownRegionsDissolution(const cv::Mat& src,
                                             int kernelSize = 3,
                                             int maxIter = 1000);

    /**
     * Remove grey regions that are not connected with black pixels.
     */
    static cv::Mat removeGrayIslands(const cv::Mat& src, int connectivity = 8);

    static double mapAlign(const cv::Mat &raw, cv::Mat &out, const AlignmentConfig &config);
};

#endif // MAPPREPROCESSING_H
