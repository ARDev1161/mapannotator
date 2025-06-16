#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <std_msgs/msg/string.hpp>
#include <cmath>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include "config.hpp"
#include "preparing/mappreprocessing.hpp"
#include "segmentation/segmentation.hpp"
#include "segmentation/labelmapping.hpp"
#include "mapgraph/zonegraph.hpp"
#include "pddl/pddlgenerator.h"

using namespace std::placeholders;

using mapping::ZoneGraph;
using mapping::NodePtr;

static std::string generatePddlFromMap(const cv::Mat1b &raw)
{
    SegmenterConfig cfg;
    cfg.denoiseConfig.cropPadding = 5;
    cfg.denoiseConfig.rankBinaryThreshold = 0.2;
    cfg.labelsListConfig.debug = false;

    cv::Mat raw8u;
    raw.convertTo(raw8u, CV_8UC1);

    cv::Mat aligned;
    MapPreprocessing::mapAlign(raw8u, aligned, cfg.alignmentConfig);

    auto [rank, crop] = MapPreprocessing::generateDenoisedAlone(aligned, cfg.denoiseConfig);

    cv::Mat1b binaryDilated = erodeBinary(rank, cfg.dilateConfig.kernelSize,
                                          cfg.dilateConfig.iterations);

    mapping::LabelsInfo labels;
    auto zones = segmentByGaussianThreshold(binaryDilated, labels, 50, 0.5);

    cv::Mat1i segmentation = cv::Mat::zeros(binaryDilated.size(), CV_32S);
    for (const auto &z : zones)
        segmentation.setTo(z.label, z.mask);

    ZoneGraph graph;
    buildGraph(graph, zones, segmentation, labels.centroids);

    PDDLGenerator gen(graph);
    std::ostringstream oss;
    oss << "(define (problem map_problem)\n"
        << "  (:domain map_domain)\n"
        << gen.objects()
        << gen.init("ROBOT_CUR_ZONE")
        << gen.goal("ROBOT_GOAL_ZONE")
        << ")\n";
    return oss.str();
}

class MapAnnotatorNode : public rclcpp::Node
{
public:
    MapAnnotatorNode() : Node("map_annotator_node")
    {
        std::string map_topic = this->declare_parameter("map_topic", std::string("/map"));
        std::string pddl_topic = this->declare_parameter("pddl_topic", std::string("/pddl/map"));

        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            map_topic, 10, std::bind(&MapAnnotatorNode::mapCallback, this, _1));
        pddl_pub_ = this->create_publisher<std_msgs::msg::String>(pddl_topic, 10);
    }

private:
    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        cv::Mat1b map(msg->info.height, msg->info.width);
        for (size_t y = 0; y < msg->info.height; ++y)
        {
            for (size_t x = 0; x < msg->info.width; ++x)
            {
                int8_t v = msg->data[y * msg->info.width + x];
                uint8_t val = (v == 0) ? 255 : 0; // treat unknown as obstacle
                map.at<uint8_t>(msg->info.height - 1 - y, x) = val;
            }
        }

        mapInfo.resolution = msg->info.resolution;
        mapInfo.originX = msg->info.origin.position.x;
        mapInfo.originY = msg->info.origin.position.y;
        double w = msg->info.origin.orientation.w;
        double x = msg->info.origin.orientation.x;
        double y = msg->info.origin.orientation.y;
        double z = msg->info.origin.orientation.z;
        mapInfo.theta = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
        mapInfo.width = msg->info.width;
        mapInfo.height = msg->info.height;

        std::string pddl = generatePddlFromMap(map);
        std_msgs::msg::String out;
        out.data = pddl;
        pddl_pub_->publish(out);
    }

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pddl_pub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MapAnnotatorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

