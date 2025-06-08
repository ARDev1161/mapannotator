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

/**
 * @brief Render a zone graph onto an OpenCV canvas.
 *
 * @tparam GraphT     Graph type implementing the IZoneGraph interface.
 * @param g           Input graph.
 * @param canvas      Output image; created if empty.
 * @param scale       Pixels per map unit (0.05 → 20 px per metre).
 * @param radius_px   Radius of the node circle in pixels.
 * @param drawWidths  Whether to display passage widths.
 * @param invertY     Flip the Y axis when drawing.
 */
template<typename GraphT = IZoneGraph>
void drawZoneGraph(const GraphT& g,
                   cv::Mat& canvas,
                   double scale        = 0.05,
                   int    radius_px    = 4,
                   bool   drawWidths   = false,
                   bool invertY = false)
{
    if (g.allNodes().empty()) {                 // ← 1. пустой граф
        canvas = cv::Mat::zeros(200, 200, CV_8UC3);
        cv::putText(canvas, "empty graph", {20,100},
                    cv::FONT_HERSHEY_PLAIN, 1.2, {0,0,255}, 2);
        return;
    }


    /* ---------- bounding box ------------------------------------------ */
    double minX =  1e9, minY =  1e9;
    double maxX = -1e9, maxY = -1e9;
    for (auto& n : g.allNodes()) {
        minX = std::min(minX, n->centroid().x);
        minY = std::min(minY, n->centroid().y);
        maxX = std::max(maxX, n->centroid().x);
        maxY = std::max(maxY, n->centroid().y);
    }

    /* ---------- размеры холста ---------------------------------------- */
    const int pad = 50;                                  // рамка
    int w = static_cast<int>(std::ceil((maxX - minX) * scale)) + pad * 2;
    int h = static_cast<int>(std::ceil((maxY - minY) * scale)) + pad * 2;

    w = std::max(w, 100);                                // защитный минимум
    h = std::max(h, 100);

    if (canvas.empty())
        canvas = cv::Mat::zeros(h, w, CV_8UC3);

    auto toPx = [&](const cv::Point2d& p){
        return cv::Point( int((p.x - minX) * scale) + pad,
                          int((p.y - minY) * scale) * (invertY ? -1 : 1) + pad );
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

        auto cent = n->centroid();
        std::string label = std::to_string(n->id()) +
//                            "[" + std::to_string(cent.x) + ";" +
//                            std::to_string(cent.y) + "]" +
                            ":" +n->type().info->path;
        cv::putText(canvas, label, pc + cv::Point(5,-5),
                    cv::FONT_HERSHEY_PLAIN, 0.9, {0,255,255}, 1);
    }
}

} // namespace mapping
