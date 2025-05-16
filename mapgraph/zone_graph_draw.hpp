#pragma once
/*-----------------------------------------------------------------------------
 *  zone_graph_draw.hpp
 *
 *  Quick‑and‑dirty visualizer on OpenCV canvas.
 *  WARNING: работает в экранных координатах centroids (X→‑‑►, Y↓),
 *  масштаб не меняет ориентацию осей (при необходимости инвертируйте Y).
 *
 *  Usage:
 *      cv::Mat img;
 *      mapping::drawZoneGraph(graph, img, 0.05);
 *      cv::imshow("Graph", img); cv::waitKey();
 *---------------------------------------------------------------------------*/
#include <opencv2/opencv.hpp>
#include "zonegraph.hpp"

namespace mapping {

inline cv::Scalar zoneColor(ZoneType t)
{
    switch (t)
    {
        case ZoneType::Corridor:                  return {255,200,200};
        case ZoneType::NarrowConnector:           return {255,180,180};
        case ZoneType::LivingRoomOfficeBedroom:   return {200,255,200};
        case ZoneType::StorageUtility:            return {200,200,255};
        case ZoneType::Sanitary:                  return {220,220,255};
        case ZoneType::Kitchenette:               return {200,255,255};
        case ZoneType::HallVestibule:             return {255,255,200};
        case ZoneType::AtriumLobby:               return {255,240,180};
        case ZoneType::Staircase:                 return {230,200,255};
        case ZoneType::ElevatorZone:              return {230,200,255};
        default:                                  return {240,240,240};
    }
}

/**
 * @param g            граф
 * @param canvas       выходное изображение (создаётся, если пустое)
 * @param scale        пикселей на единицу координат (0.05 = 20 px на 1 м)
 * @param radius_px    радиус кружка‑узла в px
 */
template<typename GraphT = IZoneGraph>
void drawZoneGraph(const GraphT& g,
                   cv::Mat& canvas,
                   double scale        = 0.05,
                   int    radius_px    = 4,
                   bool   drawWidths   = false)
{
    /* вычислим границы ---------------------------------------------------- */
    double minX=1e9, minY=1e9, maxX=-1e9, maxY=-1e9;
    for (auto& n : g.allNodes())
    {
        minX = std::min(minX, n->centroid().x);
        minY = std::min(minY, n->centroid().y);
        maxX = std::max(maxX, n->centroid().x);
        maxY = std::max(maxY, n->centroid().y);
    }
    int w = static_cast<int>((maxX-minX)*scale) + 100;
    int h = static_cast<int>((maxY-minY)*scale) + 100;
    if (canvas.empty()) canvas = cv::Mat::zeros(h, w, CV_8UC3);

    auto toPx = [&](const Point2d& p){
        return cv::Point( int((p.x - minX)*scale)+50,
                          int((p.y - minY)*scale)+50 );
    };

    /* --- draw edges ------------------------------------------------------ */
    for (auto& n : g.allNodes())
    {
        cv::Point pa = toPx(n->centroid());
        for (const auto& p : n->neighbours())
        {
            auto nb = p.neighbour.lock();
            if (!nb) continue;
            if (nb->id() < n->id()) continue;          // один раз
            cv::Point pb = toPx(nb->centroid());
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

    /* --- draw nodes ------------------------------------------------------ */
    for (auto& n : g.allNodes())
    {
        cv::Point pc = toPx(n->centroid());
        cv::circle(canvas, pc, radius_px,
                   zoneColor(n->type()), cv::FILLED, cv::LINE_AA);
        cv::circle(canvas, pc, radius_px, {50,50,50}, 1, cv::LINE_AA);

        std::string label = std::to_string(n->id()) + ":" + zoneTypeName(n->type());
        cv::putText(canvas, label, pc + cv::Point(5,-5),
                    cv::FONT_HERSHEY_PLAIN, 0.9, {0,255,255}, 1);
    }
}

} // namespace mapping
