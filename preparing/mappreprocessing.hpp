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
    /**
     * @brief Find orientation adjustment by analysing map boundaries.
     *
     * @param im     Input image (8U or 32F).
     * @param angle  Output detected angle in degrees.
     * @return       Bounding rotated rectangle of the detected wall contour.
     */
    static cv::RotatedRect findMapOrientationAdjustment(const cv::Mat& im, double& angle);

    /**
     * @brief Rotate image by a given angle.
     *
     * @param image  Source image.
     * @param angle  Rotation angle in degrees (positive = counter-clockwise).
     * @return       Rotated image with preserved background colour.
     */
    static cv::Mat rotateImage(const cv::Mat& image, double angle);

    /**
     * @brief Draw bounding box on the image.
     *
     * @param im     Source grayscale/BGR image.
     * @param rect   Rotated rectangle to render.
     * @param color  Box colour (BGR).
     * @return       Image with the box overlay.
     */
    static cv::Mat drawBB(const cv::Mat& im, const cv::RotatedRect& rect, const cv::Scalar& color = cv::Scalar(0, 0, 255));

    /**
     * @brief Draw a line on the image.
     *
     * @param im              Image to draw on.
     * @param x0,y0,x1,y1     Endpoints in pixels.
     * @param color           Line colour (BGR).
     * @param lineThickness   Line thickness in pixels.
     */
    static void drawLine(cv::Mat& im, int x0, int y0, int x1, int y1,
                         const cv::Scalar& color = cv::Scalar(255, 0, 0), int lineThickness = 1);

    /**
     * @brief Compute weighted angle histogram for the given image.
     *
     * @param rank             Binary/float map.
     * @param alignmentConfig  Alignment parameters.
     * @param debug            Dump intermediate images when true.
     * @return                 Histogram of wall angles (degrees).
     */
    static std::vector<double> computeWeightedAngleHistogram(const cv::Mat& rank,
                                                               const AlignmentConfig& alignmentConfig,
                                                               bool debug = false);

    /**
     * @brief Find best alignment angle from histogram using moving average.
     *
     * @param hist       Histogram values (degrees).
     * @param histConfig Smoothing parameters.
     * @return           Dominant angle estimate in degrees.
     */
    static double findBestAlignmentAngleFromHistogram(const std::vector<double>& hist,
                                                      const HistogramConfig& histConfig);

    /**
     * @brief Full pipeline that computes the alignment angle of a grayscale image.
     *
     * @param grayscale  Input grayscale map.
     * @param config     Alignment configuration.
     * @return           Angle in degrees to align the map.
     */
    static double findAlignmentAngle(const cv::Mat& grayscale, const AlignmentConfig& config);

    /**
     * @brief Generate a denoised map and cropping info using Segmentation helpers.
     *
     * @param raw            Input grayscale map (8U/32F).
     * @param config         Denoising parameters.
     * @param mapResolution  Map resolution (meters per pixel) for area thresholds.
     * @return               Pair of denoised binary map and cropping offsets.
     */
    static std::pair<cv::Mat, Segmentation::CropInfo> generateDenoisedAlone(const cv::Mat& raw,
                                                                            const DenoiseConfig& config,
                                                                            double mapResolution = 0.0);

    /**
     * Resolve unknown (grey) regions by snapping each pixel to the nearest
     * known class (black/white) using distance transform.
     */
    static cv::Mat unknownRegionsDissolution(const cv::Mat& src,
                                             int kernelSize = 3,
                                             int maxIter = 1000);

    /**
     * Remove grey regions that are not connected with black pixels.
     */
    static cv::Mat removeGrayIslands(const cv::Mat& src, int connectivity = 8);

    /**
     * Remove small obstacle components from a binary map.
     *
     * @param src            Binary map (obstacles are black by default).
     * @param maxDiameterPx  Remove components with area <= area of this diameter.
     * @param isInverted     If true, obstacles are white and free space is black.
     */
    static cv::Mat removeSmallBlackObjects(const cv::Mat& src, double maxDiameterPx, bool isInverted = false);

    /**
     * @brief Align map and return applied rotation.
     *
     * @param raw     Source map (8-bit grayscale).
     * @param out     Rotated output (empty if alignment disabled).
     * @param config  Alignment parameters.
     * @return        Applied rotation angle in degrees.
     */
    static double mapAlign(const cv::Mat &raw, cv::Mat &out, const AlignmentConfig &config);
};

#endif // MAPPREPROCESSING_H
