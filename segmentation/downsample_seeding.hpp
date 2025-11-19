#ifndef DOWNSAMPLE_SEEDING_HPP
#define DOWNSAMPLE_SEEDING_HPP

#include <vector>
#include <opencv2/opencv.hpp>

#include "labelmapping.hpp"

/** Parameters controlling the iterative down-sampling based seeding. */
struct DownsampleSeedsConfig {
    double sigmaStart = 1.0;     ///< initial Gaussian sigma
    double sigmaStep  = 0.5;     ///< sigma increment per iteration
    int    maxIter    = 60;      ///< maximum number of iterations
    double threshold  = 0.55;    ///< threshold applied after blur (0..1)
    int    backgroundKernel = 5; ///< erosion kernel for background mask
};

/**
 * @brief Generate robust seed masks by iteratively smoothing the free space
 *        and tracking splits between successive scales.
 *
 * @param srcBinary  Binary free-space map (0 = obstacle, 255 = free).
 * @param cfg        Down-sampling parameters.
 * @return           Vector of disjoint ZoneMask seeds (may be empty if the
 *                   input map has no free cells).
 */
std::vector<ZoneMask>
generateDownsampleSeeds(const cv::Mat1b &srcBinary,
                        const DownsampleSeedsConfig &cfg);

#endif // DOWNSAMPLE_SEEDING_HPP
