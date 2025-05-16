/*-----------------------------------------------------------------------------
 *  zone_graph.cpp
 *---------------------------------------------------------------------------*/
#include "zonegraph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mapping {

/* ===== ZoneNode =========================================================== */
ZoneNode::ZoneNode(ZoneId id,
                   ZoneType type,
                   double area,
                   double perimeter,
                   Point2d centroid,
                   std::vector<std::vector<Point2d>> contours,
                   std::optional<AABB> bbox,
                   std::string label)
    : id_{id}
    , type_{type}
    , area_{area}
    , perimeter_{perimeter}
    , centroid_{centroid}
    , contours_{std::move(contours)}
    , bbox_{std::move(bbox)}
    , label_{std::move(label)}
{}

void ZoneNode::addPassage(const NodePtr& neighbour, double width_m)
{
    /* search existing entry */
    for (auto& p : passages_)
    {
        auto sp = p.neighbour.lock();
        if (sp && sp->id() == neighbour->id())
        {
            p.widths_m.push_back(width_m);
            return;
        }
    }
    passages_.push_back({neighbour, {width_m}});
}

void ZoneNode::removePassageTo(const NodePtr& neighbour, double width_m)
{
    auto isClose = [](double a, double b, double eps=1e-5){
        return std::fabs(a-b)<=eps;
    };

    passages_.erase(
        std::remove_if(passages_.begin(), passages_.end(),
            [&](Passage& p)
            {
                auto sp = p.neighbour.lock();
                if (!sp || sp->id() != neighbour->id()) return false;

                /* remove only the width requested; NaN  => remove all */
                if (std::isnan(width_m))
                {
                    return true; // erase whole passage
                }

                p.widths_m.erase(std::remove_if(p.widths_m.begin(), p.widths_m.end(),
                                               [&](double w){ return isClose(w,width_m); }),
                                 p.widths_m.end());
                return p.widths_m.empty();

            }),
        passages_.end());
}

/* ===== ZoneGraph ========================================================== */
NodePtr ZoneGraph::addNode(ZoneType type,
                           double area,
                           double perimeter,
                           Point2d centroid,
                           std::vector<std::vector<Point2d>> contours,
                           std::optional<AABB> bbox,
                           std::string label)
{
    auto node = std::make_shared<ZoneNode>(nextId_++, type, area, perimeter,
                                           centroid, std::move(contours),
                                           std::move(bbox), std::move(label));
    nodes_.emplace(node->id(), node);
    return node;
}

bool ZoneGraph::removeNode(ZoneId id)
{
    auto it = nodes_.find(id);
    if (it == nodes_.end()) return false;

    NodePtr victim = it->second;

    /* remove references from all other nodes */
    for (auto& kv : nodes_)
        kv.second->removePassageTo(victim, std::numeric_limits<double>::quiet_NaN());

    nodes_.erase(it);
    return true;
}

bool ZoneGraph::connectZones(ZoneId a, ZoneId b, double width_m)
{
    auto na = getNode(a);
    auto nb = getNode(b);
    if (!na || !nb) return false;

    na->addPassage(nb, width_m);
    nb->addPassage(na, width_m);
    return true;
}

bool ZoneGraph::disconnectZones(ZoneId a, ZoneId b, double width_m)
{
    auto na = getNode(a);
    auto nb = getNode(b);
    if (!na || !nb) return false;

    na->removePassageTo(nb, width_m);
    nb->removePassageTo(na, width_m);
    return true;
}

NodePtr ZoneGraph::getNode(ZoneId id) const
{
    auto it = nodes_.find(id);
    return (it == nodes_.end()) ? nullptr : it->second;
}

std::vector<NodePtr> ZoneGraph::allNodes() const
{
    std::vector<NodePtr> vec;
    vec.reserve(nodes_.size());
    for (auto& kv : nodes_) vec.push_back(kv.second);
    return vec;
}

/* ===== helpers ============================================================ */
bool ZoneGraph::equalDouble(double a, double b, double eps)
{
    return std::fabs(a - b) <= eps;
}

} // namespace mapping
