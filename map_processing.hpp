#ifndef MAP_PROCESSING_HPP
#define MAP_PROCESSING_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>

#include "mapgraph/zonegraph.hpp"
#include "segmentation/segmentation.hpp"
#include "segmentation/labelmapping.hpp"

std::vector<ZoneMask>
segmentByGaussianThreshold(const cv::Mat1b &srcBinary,
                           LabelsInfo &labelsOut,
                           int maxIter = 40,
                           double sigmaStep = 0.25,
                           double threshold = 0.5);

void buildGraph(mapping::ZoneGraph &graphOut,
                std::vector<ZoneMask> zones,
                cv::Mat1i zonesMat,
                std::unordered_map<int, cv::Point> centroids);

#endif // MAP_PROCESSING_HPP

