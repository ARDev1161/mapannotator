#ifndef MAP_PROCESSING_HPP
#define MAP_PROCESSING_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>

#include "mapgraph/zonegraph.hpp"
#include "segmentation/segmentation.hpp"
#include "segmentation/labelmapping.hpp"
#include "segmentation/downsample_seeding.hpp"

/**
 * @brief Tunables for the Gaussian threshold segmentation pipeline.
 */
struct SegmentationParams {
    /** Maximum number of iterations for the legacy Gaussian loop. */
    int legacyMaxIter = 40;
    /** Sigma increment for each legacy iteration. */
    double legacySigmaStep = 0.25;
    /** Threshold used after Gaussian blur (0..1 scale). */
    double legacyThreshold = 0.5;

    double seedClearancePx = 0.0; ///< minimal distance from seeds to obstacles (pixels)

    bool useDownsampleSeeds = true; ///< prefer downsampled seeds over legacy loop
    DownsampleSeedsConfig downsampleConfig;
};

/**
 * @brief Segment free space into zones using Gaussian thresholding or downsampled seeds.
 *
 * @param srcBinary  Binary free-space map (0=wall, 1=free).
 * @param labelsOut  Output centroid map describing seed labels and positions.
 * @param params     Segmentation parameters.
 * @return           Vector of ZoneMask objects describing each segmented zone.
 */
std::vector<ZoneMask>
segmentByGaussianThreshold(const cv::Mat1b &srcBinary,
                           LabelsInfo &labelsOut,
                           const SegmentationParams &params);

/**
 * @brief Build a ZoneGraph from segmented zones and their label raster.
 *
 * @param graphOut   Resulting graph.
 * @param zones      Zone masks (consumed by value).
 * @param zonesMat   Label image matching the zones.
 * @param mapParams  Map metadata describing resolution/origin.
 * @param centroids  Centroid locations in pixel coordinates.
 */
void buildGraph(mapping::ZoneGraph &graphOut,
                std::vector<ZoneMask> zones,
                cv::Mat1i zonesMat,
                const MapInfo & mapParams,
                std::unordered_map<int, cv::Point> centroids);

/**
 * @brief Tint zones over the cropped base image for visualization.
 *
 * @param zones      Zone masks to draw.
 * @param baseImage  Source image matching the crop coordinates.
 * @param cropInfo   Crop offsets produced by denoising.
 * @param alpha      Blend factor (0..1) for zone tint.
 * @return           BGR image of size cropInfo ROI with zone overlay applied.
 */
cv::Mat renderZonesOverlay(const std::vector<ZoneMask> &zones,
                           const cv::Mat &baseImage,
                           const Segmentation::CropInfo &cropInfo,
                           double alpha = 0.65);

#endif // MAP_PROCESSING_HPP
