#pragma once
/*-----------------------------------------------------------------------------
 *  zone_graph.hpp
 *
 *  A lightweight, SOLID‑compliant graph describing connectivity between
 *  segmented indoor zones.  C++17, no external dependencies.
 *---------------------------------------------------------------------------*/
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <optional>

#include "zoneconfig.hpp"

namespace mapping {

/* ---------- basic geometry ------------------------------------------------ */
struct Point2d
{
    double x{0.0};
    double y{0.0};
};

struct AABB   // axis‑aligned bounding box
{
    Point2d min, max;
    [[nodiscard]] bool contains(const Point2d& p) const noexcept
    {
        return p.x >= min.x && p.x <= max.x &&
               p.y >= min.y && p.y <= max.y;
    }
};

/* ---------- forward declarations ------------------------------------------ */
class IZoneNode;
class ZoneNode;                         // concrete class
using ZoneId      = std::uint64_t;
using NodePtr     = std::shared_ptr<ZoneNode>;
using WeakNodePtr = std::weak_ptr<ZoneNode>;

/* ---------- passage descriptor -------------------------------------------- */
struct Passage
{
    WeakNodePtr          neighbour;  ///< target zone (weak to break cycles)
    std::vector<double>  widths_m;   ///< several doorways possible
};

/* ---------- node interface ------------------------------------------------ */
class IZoneNode
{
public:
    virtual ~IZoneNode() = default;

    virtual ZoneId                                    id()         const noexcept = 0;
    virtual ZoneType                                  type()       const noexcept = 0;
    virtual const std::vector<Passage>&               neighbours() const noexcept = 0;
    virtual double                                    area()       const noexcept = 0;
    virtual double                                    perimeter()  const noexcept = 0;
    virtual const Point2d&                            centroid()   const noexcept = 0;
    virtual const std::vector<std::vector<Point2d>>&  contours()   const noexcept = 0;
};

/* ---------- graph interface ------------------------------------------------ */
class IZoneGraph
{
public:
    virtual ~IZoneGraph() = default;

    /** Create node and return shared_ptr. */
    virtual NodePtr addNode(ZoneType type,
                            double area,
                            double perimeter,
                            Point2d centroid,
                            std::vector<std::vector<Point2d>> contours,
                            std::optional<AABB> bbox = std::nullopt,
                            std::string label = {}) = 0;

    /** Remove node and all passages pointing to it. */
    virtual bool    removeNode(ZoneId id) = 0;

    /** Undirected connection helpers. */
    virtual bool    connectZones   (ZoneId a, ZoneId b, double width_m) = 0;
    virtual bool    disconnectZones(ZoneId a, ZoneId b, double width_m) = 0;

    /* Queries -------------------------------------------------------------- */
    virtual NodePtr                 getNode(ZoneId id)           const = 0;
    virtual std::vector<NodePtr>    allNodes()                   const = 0;
};

/* ---------- concrete node -------------------------------------------------- */
class ZoneNode final : public IZoneNode,
                       public std::enable_shared_from_this<ZoneNode>
{
public:
    ZoneNode(ZoneId id,
             ZoneType type,
             double area,
             double perimeter,
             Point2d centroid,
             std::vector<std::vector<Point2d>> contours,
             std::optional<AABB> bbox,
             std::string label);

    /* IZoneNode getters implementation */
    ZoneId                                   id()         const noexcept override { return id_; }
    ZoneType                                 type()       const noexcept override { return type_; }
    const std::vector<Passage>&              neighbours() const noexcept override { return passages_; }
    double                                   area()       const noexcept override { return area_; }
    double                                   perimeter()  const noexcept override { return perimeter_; }
    const Point2d&                           centroid()   const noexcept override { return centroid_; }
    const std::vector<std::vector<Point2d>>& contours()   const noexcept override { return contours_; }

    /* internal for ZoneGraph */
    void addPassage(const NodePtr& neighbour, double width_m);
    void removePassageTo(const NodePtr& neighbour, double width_m);

private:
    ZoneId                                   id_{0};
    ZoneType                                 type_{ZoneType::Unknown};
    double                                   area_{0.0};
    double                                   perimeter_{0.0};
    Point2d                                  centroid_{};
    std::vector<std::vector<Point2d>>        contours_;
    std::optional<AABB>                      bbox_;
    std::string                              label_;
    std::vector<Passage>                     passages_;
};

/* ---------- concrete graph ------------------------------------------------- */
class ZoneGraph final : public IZoneGraph
{
public:
    NodePtr addNode(ZoneType type,
                    double area,
                    double perimeter,
                    Point2d centroid,
                    std::vector<std::vector<Point2d>> contours,
                    std::optional<AABB> bbox = std::nullopt,
                    std::string label = {}) override;

    bool    removeNode(ZoneId id) override;

    bool    connectZones   (ZoneId a, ZoneId b, double width_m) override;
    bool    disconnectZones(ZoneId a, ZoneId b, double width_m) override;

    NodePtr                 getNode(ZoneId id) const override;
    std::vector<NodePtr>    allNodes()          const override;

private:
    static bool equalDouble(double a, double b, double eps = 1e-5);

    ZoneId                                   nextId_{1};
    std::unordered_map<ZoneId, NodePtr>      nodes_;
};

} // namespace mapping
