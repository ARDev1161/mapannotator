#ifndef MAP_PROCESSING_HPP
#define MAP_PROCESSING_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>

#include "mapgraph/zonegraph.hpp"
#include "segmentation/segmentation.hpp"
#include "segmentation/labelmapping.hpp"
#include "segmentation/downsample_seeding.hpp"

struct SegmentationParams {
    int legacyMaxIter = 40;
    double legacySigmaStep = 0.25;
    double legacyThreshold = 0.5;

    bool useDownsampleSeeds = true;
    DownsampleSeedsConfig downsampleConfig;
};

std::vector<ZoneMask>
segmentByGaussianThreshold(const cv::Mat1b &srcBinary,
                           LabelsInfo &labelsOut,
                           const SegmentationParams &params);

void buildGraph(mapping::ZoneGraph &graphOut,
                std::vector<ZoneMask> zones,
                cv::Mat1i zonesMat,
                const MapInfo & mapParams,
                std::unordered_map<int, cv::Point> centroids);

cv::Mat renderZonesOverlay(const std::vector<ZoneMask> &zones,
                           const cv::Mat &baseImage,
                           const Segmentation::CropInfo &cropInfo,
                           double alpha = 0.65);

#endif // MAP_PROCESSING_HPP
