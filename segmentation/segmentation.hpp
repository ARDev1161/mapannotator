#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <filesystem>
#include <set>
#include "../utils.hpp"

/**
 * @brief Collection of static image segmentation helpers.
 */
class Segmentation {
public:
    /** Information about cropping offsets. */
    struct CropInfo {
        int left;   ///< offset from the left
        int top;    ///< offset from the top
        int right;  ///< offset from the right
        int bottom; ///< offset from the bottom
    };

    /**
     * Find the bounding box for the given map using the background pixel (0,0)
     * as reference.
     *
     * @param map  Binary image where zero is treated as background.
     * @return     Tight bounding box of non-zero pixels (may be empty).
     */
    static cv::Rect findBB(const cv::Mat& map);

    /**
     * @brief Compute cropping info with optional padding.
     *
     * @param map      Binary map to analyse (8U, single channel).
     * @param padding  Extra pixels to retain around the bounding box.
     * @return         Offsets from each border to the cropped ROI.
     */
    static CropInfo cropSingleInfo(const cv::Mat& map, int padding = 0);
    /**
     * @brief Compute common cropping info for two maps.
     *
     * @param rankMap   First map.
     * @param trackMap  Second map (same size).
     * @return          Offsets covering non-zero areas of both maps.
     */
    static CropInfo cropBundleInfo(const cv::Mat& rankMap, const cv::Mat& trackMap);

    /**
     * @brief Crop map using previously obtained CropInfo.
     *
     * @param map   Source image.
     * @param info  CropInfo offsets (relative to original size).
     * @return      Cropped clone.
     */
    static cv::Mat cropBackground(const cv::Mat& map, const CropInfo& info);
    /**
     * @brief Restore original size by padding cropped image with background colour.
     *
     * @param map         Cropped image.
     * @param info        Offsets used during cropping.
     * @param background  Background colour to fill.
     * @return            Uncropped image.
     */
    static cv::Mat uncropBackground(const cv::Mat& map, const CropInfo& info, const cv::Scalar& background);

    /** Replace pixels with the specified border label by 255.
     * If border_label == -1 the maximal label value is used. */
    static cv::Mat remapBorder(const cv::Mat& map, int border_label = -1);

    /**
     * @brief Remove small connected components from a binary image.
     *
     * @param src           Binary image (0/255).
     * @param method        "fixed" or "relative" size criterion.
     * @param min_size      Minimum component size (px or fraction).
     * @param connectivity  4 or 8 connectivity.
     * @param debug         Dump debug output when true.
     * @return              Cleaned binary image.
     */
    static cv::Mat removeSmallConnectedComponents(const cv::Mat& src, const std::string& method = "fixed",
                                                    int min_size = 1, int connectivity = 4, bool debug = false);

    /**
     * @brief Generate watershed seeds using Gaussian blur.
     *
     * @param src              Source map (8U).
     * @param background_mask  Mask defining background (0 where background).
     * @param sigma            Gaussian sigma.
     * @param threshold        Threshold after blur (0..1).
     * @return                 Seed image (binary).
     */
    static cv::Mat generateGaussianSeeds(const cv::Mat& src, const cv::Mat& background_mask,
                                           double sigma, double threshold = 0.05);

    /**
     * @brief Perform watershed segmentation.
     *
     * @param src      3-channel source image.
     * @param labels   Initial markers (CV_32S).
     * @param connectivity 4 or 8 connectivity for flooding.
     * @return         Label image (CV_32S) where -1 marks watershed lines.
     */
    static cv::Mat createWatershedSegment(const cv::Mat& src, const cv::Mat& labels, int connectivity = 4);
};

#endif // SEGMENTATION_H
