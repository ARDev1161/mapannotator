#pragma once
/*-----------------------------------------------------------------------------
 *  zone_graph_dot.hpp
 *
 *  Helper: dump ZoneGraph into Graphviz DOT syntax.
 *  Usage:
 *      #include "zone_graph_dot.hpp"
 *      std::ofstream ofs("graph.dot");
 *      mapping::writeDot(graph, ofs);
 *      // затем   dot -Tpng graph.dot -o graph.png
 *---------------------------------------------------------------------------*/
#include <ostream>
#include <unordered_set>
#include <sstream>
#include <iomanip>
#include "zonegraph.hpp"   //  IZoneGraph / ZoneType

namespace mapping {


/* --- main writer --------------------------------------------------------- */
/**
 * @brief Serialise a ZoneGraph into Graphviz DOT format.
 *
 * @tparam GraphT  Graph type implementing the IZoneGraph interface.
 * @param g        Graph instance to serialise.
 * @param os       Output stream.
 * @param withWidths If true, passage widths are added as edge labels.
 */
template<typename GraphT = IZoneGraph>
void writeDot(const GraphT& g, std::ostream& os, bool withWidths = true)
{
    os << "graph ZoneConnectivity {\n";
    os << "  node [shape=ellipse, style=filled, fillcolor=\"#e0f7ff\"];\n";

    /* --- nodes ----------------------------------------------------------- */
    for (auto& n : g.allNodes())
    {
        os << "  " << n->id()
           << " [label=\"" << n->id() << "\\n"
           << n->type().info->path << "\\n"
           << std::fixed << std::setprecision(1)
           << n->area() << "m2\"];\n";
    }

    /* --- edges (avoid duplicates) --------------------------------------- */
    std::unordered_set<std::uint64_t> seen;

    auto makeKey = [](ZoneId a, ZoneId b)->std::uint64_t
    {
        return (static_cast<std::uint64_t>(std::min(a,b))<<32) |
                static_cast<std::uint64_t>(std::max(a,b));
    };

    for (auto& n : g.allNodes())
    {
        ZoneId idA = n->id();
        for (const auto& p : n->neighbours())
        {
            auto nb = p.neighbour.lock();
            if (!nb) continue;
            ZoneId idB = nb->id();
            std::uint64_t key = makeKey(idA,idB);
            if (seen.count(key)) continue;
            seen.insert(key);

            os << "  " << idA << " -- " << idB;
            if (withWidths)
            {
                std::ostringstream lbl;
                for (size_t i=0;i<p.widths_m.size();++i)
                {
                    if (i) lbl << ", ";
                    lbl << std::fixed << std::setprecision(2) << p.widths_m[i] << "m";
                }
                os << " [label=\"" << lbl.str() << "\"]";
            }
            os << ";\n";
        }
    }
    os << "}\n";
}

} // namespace mapping
