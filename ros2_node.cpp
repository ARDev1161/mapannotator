#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cmath>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include "visualization.hpp"
#include "config.hpp"
#include "preparing/mappreprocessing.hpp"
#include "segmentation/segmentation.hpp"
#include "segmentation/labelmapping.hpp"
#include "mapgraph/zonegraph.hpp"
#include "pddl/pddlgenerator.h"
#include "map_processing.hpp"

using namespace std::placeholders;

using mapping::ZoneGraph;
using mapping::NodePtr;




static std::string generatePddlFromMap(const cv::Mat1b &raw,
                                       const SegmenterConfig &cfg,
                                       const SegmentationParams &seg_params,
                                       const std::string &start_zone,
                                       const std::string &goal_zone,
                                       cv::Mat *vis_out = nullptr)
{
    cv::Mat raw8u;
    raw.convertTo(raw8u, CV_8UC1);

    cv::Mat aligned;
    MapPreprocessing::mapAlign(raw8u, aligned, cfg.alignmentConfig);

    auto [rank, crop] = MapPreprocessing::generateDenoisedAlone(aligned, cfg.denoiseConfig);

    cv::Mat1b binaryDilated = erodeBinary(rank, cfg.dilateConfig.kernelSize,
                                          cfg.dilateConfig.iterations);

    LabelsInfo labels;
    auto zones = segmentByGaussianThreshold(binaryDilated, labels, seg_params);

    cv::Mat1i segmentation = cv::Mat::zeros(binaryDilated.size(), CV_32S);
    for (const auto &z : zones)
        segmentation.setTo(z.label, z.mask);

    ZoneGraph graph;
    buildGraph(graph, zones, segmentation, mapInfo, labels.centroids);

    cv::Mat baseBinaryFull = Segmentation::uncropBackground(binaryDilated, crop, cv::Scalar(0));
    cv::Mat vis = renderZonesOverlay(zones, baseBinaryFull, crop, 0.65);
    mapping::drawZoneGraphOnMap(graph, vis, mapInfo);
    if (vis_out) {
        *vis_out = vis.clone();
    }

    PDDLGenerator gen(graph);
    std::ostringstream oss;
    oss << "(define (problem map_problem)\n"
        << "  (:domain map_domain)\n"
        << gen.objects()
        << gen.init(start_zone)
        << gen.goal(goal_zone)
        << ")\n";
    return oss.str();
}

class MapAnnotatorNode : public rclcpp::Node
{
public:
    MapAnnotatorNode() : Node("map_annotator_node")
    {
        map_topic_  = this->declare_parameter("map_topic", std::string("/map"));
        pddl_topic_ = this->declare_parameter("pddl_topic", std::string("/pddl/map"));
        segmentation_topic_ = this->declare_parameter("segmentation_topic", std::string("/mapannotator/segmentation"));

        config_.denoiseConfig.cropPadding =
            this->declare_parameter("denoise.crop_padding", 5);
        config_.denoiseConfig.rankBinaryThreshold =
            this->declare_parameter("denoise.rank_binary_threshold", 0.2);
        config_.dilateConfig.kernelSize =
            this->declare_parameter("dilate.kernel_size", 3);
        config_.dilateConfig.iterations =
            this->declare_parameter("dilate.iterations", 1);
        config_.alignmentConfig.enable =
            this->declare_parameter("alignment.enable", true);

        seg_params_.legacyMaxIter = this->declare_parameter("segmentation.max_iter", 50);
        seg_params_.legacySigmaStep = this->declare_parameter("segmentation.sigma_step", 0.5);
        seg_params_.legacyThreshold = this->declare_parameter("segmentation.threshold", 0.5);
        seg_params_.downsampleConfig.maxIter = seg_params_.legacyMaxIter;
        seg_params_.downsampleConfig.sigmaStep = seg_params_.legacySigmaStep;
        seg_params_.downsampleConfig.threshold = seg_params_.legacyThreshold;
        seg_params_.downsampleConfig.sigmaStart =
            this->declare_parameter("segmentation.downsample_sigma_start", 1.0);
        seg_params_.downsampleConfig.backgroundKernel =
            this->declare_parameter("segmentation.background_kernel", 5);
        seg_params_.useDownsampleSeeds =
            this->declare_parameter("segmentation.use_downsample_seeds", true);

        start_zone_ = this->declare_parameter("start_zone", std::string("ROBOT_CUR_ZONE"));
        goal_zone_  = this->declare_parameter("goal_zone", std::string("ROBOT_GOAL_ZONE"));

        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            map_topic_, 10, std::bind(&MapAnnotatorNode::mapCallback, this, _1));
        pddl_pub_ = this->create_publisher<std_msgs::msg::String>(pddl_topic_, 10);
        segmentation_pub_ = this->create_publisher<sensor_msgs::msg::Image>(segmentation_topic_, 10);
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

    cv::Mat vis;
    std::string pddl = generatePddlFromMap(map, config_,
                                          seg_params_,
                                          start_zone_, goal_zone_, &vis);

    if(!isHeadlessMode())
    {
        if(!isHeadlessMode())
        {
            cv::imshow("segmented", vis);
        }
    }

        std_msgs::msg::String out;
        out.data = pddl;
        pddl_pub_->publish(out);

        sensor_msgs::msg::Image img_msg;
        img_msg.header = msg->header;
        img_msg.height = vis.rows;
        img_msg.width = vis.cols;
        img_msg.encoding = "bgr8";
        img_msg.step = static_cast<sensor_msgs::msg::Image::_step_type>(vis.step);
        img_msg.data.assign(vis.datastart, vis.dataend);
        segmentation_pub_->publish(img_msg);
    }

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pddl_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr segmentation_pub_;
    std::string map_topic_;
    std::string pddl_topic_;
    std::string segmentation_topic_;
    SegmenterConfig config_;
    SegmentationParams seg_params_;
    std::string start_zone_;
    std::string goal_zone_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MapAnnotatorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
