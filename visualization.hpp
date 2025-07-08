#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include "mapgraph/zonegraph.hpp"
#include "mapgraph/typeregistry.h"
#include "utils.hpp"

inline cv::Mat3b colorizeSegmentation(const cv::Mat1i& seg,
                                      const cv::Mat1b& wallMask,
                                      const std::string& winName = "",
                                      const std::string& pngPath = "",
                                      int colormap = cv::COLORMAP_JET)
{
    CV_Assert(!seg.empty() && seg.type() == CV_32S);

    double minVal, maxVal;
    cv::minMaxLoc(seg, &minVal, &maxVal);
    double scale = (maxVal > 0) ? 255.0 / maxVal : 1.0;

    cv::Mat1b seg8u;
    seg.convertTo(seg8u, CV_8U, scale);

    cv::Mat3b color;
    cv::applyColorMap(seg8u, color, colormap);

    color.setTo(cv::Vec3b(50,50,50), seg8u == 0);

    double alpha = 0.9;
    cv::Scalar wallColor(0,0,0);
    cv::Mat overlay(color.size(), color.type(), wallColor);
    cv::Mat blended = color.clone();
    overlay.copyTo(blended, wallMask);
    cv::addWeighted(color, 1.0, blended, alpha, 0.0, blended);

    if (!winName.empty()) {
        showMat(winName, blended);
        cv::waitKey(0);
    }
    if (!pngPath.empty())
        cv::imwrite(pngPath, blended);

    return blended;
}

namespace mapping {

template<typename GraphT = IZoneGraph>
inline void drawZoneGraphOnMap(const GraphT& g,
                               cv::Mat& canvas,
                               const MapInfo& info,
                               int radius_px = 4,
                               bool drawWidths = false)
{
    for (auto& n : g.allNodes())
    {
        cv::Point pa = worldToPixel(n->centroid(), info);
        for (const auto& p : n->neighbours())
        {
            auto nb = p.neighbour.lock();
            if (!nb) continue;
            if (nb->id() < n->id()) continue;
            cv::Point pb = worldToPixel(nb->centroid(), info);
            cv::line(canvas, pa, pb, {200,200,200}, 1, cv::LINE_AA);
            if (drawWidths && !p.widths_m.empty())
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(1) << p.widths_m[0];
                cv::Point mid = (pa + pb) * 0.5;
                cv::putText(canvas, ss.str(), mid,
                            cv::FONT_HERSHEY_PLAIN, 0.8, {255,50,50}, 1);
            }
        }
    }

    for (auto& n : g.allNodes())
    {
        cv::Point pc = worldToPixel(n->centroid(), info);
        cv::circle(canvas, pc, radius_px,
                   zoneColor(n->type()), cv::FILLED, cv::LINE_AA);
        cv::circle(canvas, pc, radius_px, {50,50,50}, 1, cv::LINE_AA);
        std::string label = std::to_string(n->id()) + ":" + n->type().info->path;
        cv::putText(canvas, label, pc + cv::Point(5,-5),
                    cv::FONT_HERSHEY_PLAIN, 0.9, {0,255,255}, 1);
    }
}

} // namespace mapping

