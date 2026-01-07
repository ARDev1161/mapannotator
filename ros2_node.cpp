#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cmath>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include "visualization.hpp"
#include "config.hpp"
#include "preparing/mappreprocessing.hpp"
#include "segmentation/segmentation.hpp"
#include "segmentation/labelmapping.hpp"
#include "mapgraph/zonegraph.hpp"
#include "mapgraph/typeregistry.h"
#include "pddl/pddlgenerator.h"
#include "map_processing.hpp"

using namespace std::placeholders;

using mapping::ZoneGraph;
using mapping::NodePtr;

struct AgentZoneInfo
{
    std::uint32_t id{0};
    std::uint32_t type_id{0};
    std::string name;
    std::string type;
    cv::Point2d centroid;
    std::array<std::uint8_t, 3> color{{0, 0, 0}}; // RGB
    std::vector<cv::Point2d> chain_code;
};

static int normalizeChainMethod(int method)
{
    switch (method) {
        case cv::CHAIN_APPROX_NONE:
        case cv::CHAIN_APPROX_SIMPLE:
        case cv::CHAIN_APPROX_TC89_L1:
        case cv::CHAIN_APPROX_TC89_KCOS:
            return method;
        default:
            return cv::CHAIN_APPROX_SIMPLE;
    }
}

static int parseChainMethod(const std::string &method, rclcpp::Logger logger)
{
    std::string lowered = method;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lowered == "none")
        return cv::CHAIN_APPROX_NONE;
    if (lowered == "simple")
        return cv::CHAIN_APPROX_SIMPLE;
    if (lowered == "tc89_l1")
        return cv::CHAIN_APPROX_TC89_L1;
    if (lowered == "tc89_kcos")
        return cv::CHAIN_APPROX_TC89_KCOS;
    RCLCPP_WARN(logger, "Unknown zones.chain_approx_method '%s', falling back to 'simple'",
                method.c_str());
    return cv::CHAIN_APPROX_SIMPLE;
}

static std::vector<cv::Point> extractLargestContour(const cv::Mat1b &mask, int chain_method)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, normalizeChainMethod(chain_method));
    if (contours.empty())
        return {};

    double max_area = 0.0;
    int max_idx = -1;
    for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
        double area = std::fabs(cv::contourArea(contours[i]));
        if (area > max_area) {
            max_area = area;
            max_idx = i;
        }
    }
    if (max_idx < 0)
        return {};
    return contours[static_cast<std::size_t>(max_idx)];
}

static std::vector<cv::Point2d> contourToWorldPoints(const std::vector<cv::Point> &contour)
{
    std::vector<cv::Point2d> out;
    if (contour.empty())
        return out;

    out.reserve(contour.size() + 1);
    for (const auto &pt : contour)
        out.push_back(pixelToWorld(pt, mapInfo));
    out.push_back(pixelToWorld(contour.front(), mapInfo)); // close contour
    return out;
}

static std::string escapeJson(const std::string &src)
{
    std::string out;
    out.reserve(src.size());
    for (char c : src) {
        if (c == '\\' || c == '"')
            out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

static std::string serializeAgentZones(const std::vector<AgentZoneInfo> &zones)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "{\"zones\":[";
    for (std::size_t i = 0; i < zones.size(); ++i) {
        const auto &z = zones[i];
        oss << "{"
            << "\"id\":" << z.id << ','
            << "\"type_id\":" << z.type_id << ','
            << "\"name\":\"" << escapeJson(z.name) << "\","
            << "\"type\":\"" << escapeJson(z.type) << "\","
            << "\"color\":{"
            << "\"r\":" << static_cast<int>(z.color[0]) << ','
            << "\"g\":" << static_cast<int>(z.color[1]) << ','
            << "\"b\":" << static_cast<int>(z.color[2]) << "},"
            << "\"centroid\":{\"x\":" << z.centroid.x << ",\"y\":" << z.centroid.y << "},"
            << "\"chain_code\":[";
        for (std::size_t j = 0; j < z.chain_code.size(); ++j) {
            const auto &pt = z.chain_code[j];
            oss << "{\"x\":" << pt.x << ",\"y\":" << pt.y << "}";
            if (j + 1 < z.chain_code.size())
                oss << ",";
        }
        oss << "],\"passages\":[]}";
        if (i + 1 < zones.size())
            oss << ",";
    }
    oss << "]}";
    return oss.str();
}

static cv::Vec3b labelColor(int label)
{
    cv::RNG rng(static_cast<uint64_t>(label) * 9781u + 13579u);
    return cv::Vec3b(rng.uniform(60, 255),
                     rng.uniform(60, 255),
                     rng.uniform(60, 255));
}

static std::vector<AgentZoneInfo> buildAgentZones(const std::vector<ZoneMask> &zones,
                                                  const mapping::ZoneGraph &graph,
                                                  const PDDLGenerator &gen,
                                                  int chain_method)
{
    std::unordered_map<int, AgentZoneInfo> by_id;

    for (const auto &z : zones) {
        auto contour = extractLargestContour(z.mask, chain_method);
        AgentZoneInfo info;
        info.id = z.label;
        if (!contour.empty())
            info.chain_code = contourToWorldPoints(contour);
        cv::Vec3b bgr = labelColor(z.label);
        info.color = {bgr[2], bgr[1], bgr[0]};
        by_id[info.id] = std::move(info);
    }

    for (const auto &node : graph.allNodes()) {
        const int id = static_cast<int>(node->id());
        auto &info = by_id[id]; // creates if missing
        info.id = static_cast<std::uint32_t>(id);
        info.type_id = static_cast<std::uint32_t>(node->type().id());
        info.name = gen.zoneLabel(node);
        info.type = node->type().path();
        info.centroid = node->centroid();
        cv::Vec3b bgr = labelColor(id);
        info.color = {bgr[2], bgr[1], bgr[0]};
    }

    std::vector<AgentZoneInfo> out;
    out.reserve(by_id.size());
    for (auto &kv : by_id)
        out.push_back(std::move(kv.second));
    std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) { return a.id < b.id; });
    return out;
}

static std::string generatePddlFromMap(const cv::Mat1b &raw,
                                       const SegmenterConfig &cfg,
                                       const SegmentationParams &seg_params,
                                       const std::string &start_zone,
                                       const std::string &goal_zone,
                                       cv::Mat *vis_out = nullptr,
                                       std::vector<AgentZoneInfo> *agent_zones_out = nullptr,
                                       int chain_method = cv::CHAIN_APPROX_SIMPLE)
{
    cv::Mat raw8u;
    raw.convertTo(raw8u, CV_8UC1);

    cv::Mat aligned;
    double alignmentAngle = MapPreprocessing::mapAlign(raw8u,
                                                       aligned,
                                                       cfg.alignmentConfig);
    if (aligned.empty())
        aligned = raw8u.clone();
    mapInfo.height = aligned.rows;
    mapInfo.width = aligned.cols;
    mapInfo.theta += alignmentAngle;

    auto [rank, crop] = MapPreprocessing::generateDenoisedAlone(aligned, cfg.denoiseConfig);
    {
        // Account for cropping when translating pixel coords to the world frame.
        int newWidth  = aligned.cols - crop.left - crop.right;
        int newHeight = aligned.rows - crop.top  - crop.bottom;
        if (newWidth > 0 && newHeight > 0) {
            double dx = crop.left   * mapInfo.resolution;
            double dy = crop.bottom * mapInfo.resolution;
            double c  = std::cos(mapInfo.theta);
            double s  = std::sin(mapInfo.theta);
            mapInfo.originX += dx * c - dy * s;
            mapInfo.originY += dx * s + dy * c;
            mapInfo.width  = rank.cols;
            mapInfo.height = rank.rows;
        }
    }

    cv::Mat1b binaryDilated = erodeBinary(rank, cfg.dilateConfig.kernelSize,
                                          cfg.dilateConfig.iterations);

    LabelsInfo labels = LabelMapping::computeLabels(binaryDilated, 0, seg_params.seedClearancePx);
    auto zones = segmentByGaussianThreshold(binaryDilated, labels, seg_params);

    cv::Mat1i segmentation = cv::Mat::zeros(binaryDilated.size(), CV_32S);
    for (const auto &z : zones)
        segmentation.setTo(z.label, z.mask);

    ZoneGraph graph;
    buildGraph(graph, zones, segmentation, mapInfo, labels.centroids);

    cv::Mat baseImage = aligned.empty() ? cv::Mat(raw.clone()) : aligned.clone();
    if (baseImage.channels() > 1)
        cv::cvtColor(baseImage, baseImage, cv::COLOR_BGR2GRAY);
    cv::Mat vis = renderZonesOverlay(zones, baseImage, crop, 0.65);
    mapping::drawZoneGraphOnMap(graph, vis, mapInfo);
    if (vis_out) {
        *vis_out = vis.clone();
    }

    PDDLGenerator gen(graph);

    if (agent_zones_out) {
        *agent_zones_out = buildAgentZones(zones, graph, gen, chain_method);
    }

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
        rc_agent_topic_ = this->declare_parameter("rc_agent_topic", std::string("/mapannotator/rc_agent_zones"));

        config_.denoiseConfig.rankBinaryThreshold =
            this->declare_parameter("denoise.rank_binary_threshold", 0.2);
        config_.dilateConfig.kernelSize =
            this->declare_parameter("dilate.kernel_size", 3);
        config_.dilateConfig.iterations =
            this->declare_parameter("dilate.iterations", 1);
        config_.alignmentConfig.enable =
            this->declare_parameter("alignment.enable", true);

        seg_params_.maxIter = this->declare_parameter("segmentation.max_iter", 50);
        seg_params_.sigmaStep = this->declare_parameter("segmentation.sigma_step", 0.5);
        seg_params_.threshold = this->declare_parameter("segmentation.threshold", 0.5);
        seg_params_.seedClearancePx =
            this->declare_parameter("segmentation.seed_clearance_px", 0.0);
        chain_approx_method_ =
            parseChainMethod(this->declare_parameter("zones.chain_approx_method",
                                                     std::string("simple")),
                             this->get_logger());
        seed_clearance_m_ =
            this->declare_parameter("segmentation.seed_clearance_m", 0.0);

        start_zone_ = this->declare_parameter("start_zone", std::string("ROBOT_CUR_ZONE"));
        goal_zone_  = this->declare_parameter("goal_zone", std::string("ROBOT_GOAL_ZONE"));

        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            map_topic_, 10, std::bind(&MapAnnotatorNode::mapCallback, this, _1));
        pddl_pub_ = this->create_publisher<std_msgs::msg::String>(pddl_topic_, 10);
        segmentation_pub_ = this->create_publisher<sensor_msgs::msg::Image>(segmentation_topic_, 10);
        rc_agent_pub_ = this->create_publisher<std_msgs::msg::String>(rc_agent_topic_, 10);
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

        if (seed_clearance_m_ > 0.0 && mapInfo.resolution > 0.0)
            seg_params_.seedClearancePx = seed_clearance_m_ / mapInfo.resolution;

        cv::Mat vis;
        std::vector<AgentZoneInfo> agent_zones;
        std::string pddl = generatePddlFromMap(map, config_,
                                              seg_params_,
                                              start_zone_, goal_zone_, &vis,
                                              &agent_zones,
                                              chain_approx_method_);

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

        publishRcAgentZones(agent_zones);
    }

    void publishRcAgentZones(const std::vector<AgentZoneInfo> &zones)
    {
        if (!rc_agent_pub_ || zones.empty())
            return;
        std_msgs::msg::String packed;
        packed.data = serializeAgentZones(zones);
        rc_agent_pub_->publish(packed);
    }

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pddl_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr segmentation_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr rc_agent_pub_;
    std::string map_topic_;
    std::string pddl_topic_;
    std::string segmentation_topic_;
    std::string rc_agent_topic_;
    SegmenterConfig config_;
    SegmentationParams seg_params_;
    std::string start_zone_;
    std::string goal_zone_;
    double seed_clearance_m_{0.0};
    int chain_approx_method_{cv::CHAIN_APPROX_SIMPLE};
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MapAnnotatorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
