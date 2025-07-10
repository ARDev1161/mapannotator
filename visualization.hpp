#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include "mapgraph/zonegraph.hpp"
#include "mapgraph/typeregistry.h"
#include "utils.hpp"

inline cv::Mat colorizeSegmentation(const cv::Mat1i& seg,
                                      const cv::Mat1b& wallMask,
                                      int colormap = cv::COLORMAP_JET)
{
    CV_Assert(!seg.empty() && seg.type() == CV_32S);

    /* 1.  нормируем [0 .. maxLabel] → [0 .. 255]  ------------------------- */
    double minVal, maxVal;
    cv::minMaxLoc(seg, &minVal, &maxVal);
    double scale = (maxVal > 0) ? 255.0 / maxVal : 1.0;

    cv::Mat1b seg8u;
    seg.convertTo(seg8u, CV_8U, scale); // CV_8U для applyColorMap

    /* 2.  применяем цветовую карту  --------------------------------------- */
    cv::Mat3b color;
    cv::applyColorMap(seg8u, color, colormap);

    /* 3.  делаем фон серым для контраста (опц.) -------------------- */
    // фон = те же пиксели, что были 0 в seg (т.е. seg8u == 0)
    color.setTo(cv::Vec3b(50,50,50), seg8u == 0);

    // добавляем стены
    double alpha = 0.9;                  // непрозрачность 0…1 (0.6 = 60 %)
    cv::Scalar wallColor(0, 0, 0);     // чёрные стены (BGR)

    /* создаём BGR-картинку только для стен */
    cv::Mat overlay(color.size(), color.type(), wallColor);

    /* смешиваем: dst = imgColor*(1-α) + overlay*α  только там, где стена */
    cv::Mat blended = color.clone();
    overlay.copyTo(blended, wallMask);   // overlay → blended (там, где стена)

    /* linearBlend = img*(1-α) + blended*α, но только по маске */
    cv::addWeighted(color, 1.0, blended, alpha, 0.0, blended);

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
            cv::line(canvas, pa, pb, {42,42,42}, 1, cv::LINE_AA);
            if (drawWidths && !p.widths_m.empty())
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(1) << p.widths_m[0];
                cv::Point mid = (pa + pb) * 0.5;
                cv::putText(canvas, ss.str(), mid,
                            cv::FONT_HERSHEY_PLAIN, 0.8, {42,0,0}, 1);
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
                    cv::FONT_HERSHEY_PLAIN, 0.9, {0,42,42}, 1);
    }
}

} // namespace mapping

