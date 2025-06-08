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
     */
    static cv::Rect findBB(const cv::Mat& map);

    /** Compute cropping info with optional padding. */
    static CropInfo cropSingleInfo(const cv::Mat& map, int padding = 0);
    /** Compute common cropping info for two maps. */
    static CropInfo cropBundleInfo(const cv::Mat& rankMap, const cv::Mat& trackMap);

    /** Crop map using previously obtained CropInfo. */
    static cv::Mat cropBackground(const cv::Mat& map, const CropInfo& info);
    /** Restore original size by padding cropped image with background colour. */
    static cv::Mat uncropBackground(const cv::Mat& map, const CropInfo& info, const cv::Scalar& background);

    /** Replace pixels with the specified border label by 255.
     * If border_label == -1 the maximal label value is used. */
    static cv::Mat remapBorder(const cv::Mat& map, int border_label = -1);

    /** Remove small connected components from a binary image. */
    static cv::Mat removeSmallConnectedComponents(const cv::Mat& src, const std::string& method = "fixed",
                                                    int min_size = 1, int connectivity = 4, bool debug = false);

    /** Generate watershed seeds using Gaussian blur. */
    static cv::Mat generateGaussianSeeds(const cv::Mat& src, const cv::Mat& background_mask,
                                           double sigma, double threshold = 0.05);

    /** Perform watershed segmentation.
     *  @param src     3-channel source image
     *  @param labels  initial markers (CV_32S)
     */
    static cv::Mat createWatershedSegment(const cv::Mat& src, const cv::Mat& labels, int connectivity = 4);
};

#endif // SEGMENTATION_H
