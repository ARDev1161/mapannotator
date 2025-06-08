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
#include <opencv2/opencv.hpp>

#include "zoneconfig.hpp"

namespace mapping {

/* ---------- basic geometry ------------------------------------------------ */
/**
 * @brief Axis-aligned bounding box helper.
 */
struct AABB
{
    cv::Point2d min;   ///< bottom-left (or top-left) corner
    cv::Point2d max;   ///< top-right corner

    /** Check whether a point lies inside the box. */
    [[nodiscard]] bool contains(const cv::Point2d& p) const noexcept
    {
        return p.x >= min.x && p.x <= max.x &&
               p.y >= min.y && p.y <= max.y;
    }

    /** Width of the box. */
    [[nodiscard]] double width () const noexcept { return max.x - min.x; }
    /** Height of the box. */
    [[nodiscard]] double height() const noexcept { return max.y - min.y; }
};

/* ---------- forward declarations ------------------------------------------ */
class IZoneNode;
class ZoneNode;                         // concrete class
using ZoneId      = std::uint64_t;
using NodePtr     = std::shared_ptr<ZoneNode>;
using WeakNodePtr = std::weak_ptr<ZoneNode>;

/* ---------- passage descriptor -------------------------------------------- */
/**
 * @brief Connection information between neighbouring zones.
 */
struct Passage
{
    WeakNodePtr          neighbour;  ///< target zone (weak to break cycles)
    std::vector<double>  widths_m;   ///< one zone may have multiple doorway widths
};

/* ---------- node interface ------------------------------------------------ */
/**
 * @brief Abstract interface for a zone node in the graph.
 */
class IZoneNode
{
public:
    virtual ~IZoneNode() = default;

    virtual ZoneId                                    id()         const noexcept = 0;
    virtual ZoneType                                  type()       const noexcept = 0;
    virtual const std::vector<Passage>&               neighbours() const noexcept = 0;
    virtual double                                    area()       const noexcept = 0;
    virtual double                                    perimeter()  const noexcept = 0;
    virtual const cv::Point2d&                            centroid()   const noexcept = 0;
    virtual const std::vector<std::vector<cv::Point2d>>&  contours()   const noexcept = 0;
};

/* ---------- graph interface ------------------------------------------------ */
class IZoneGraph
{
public:
    virtual ~IZoneGraph() = default;

    /** Create node and return shared_ptr. */
    virtual NodePtr addNode(ZoneId id,
                            ZoneType type,
                            double area,
                            double perimeter,
                            cv::Point2d centroid,
                            std::vector<std::vector<cv::Point2d>> contours,
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
/**
 * @brief Concrete implementation of IZoneNode used by ZoneGraph.
 */
class ZoneNode final : public IZoneNode,
                       public std::enable_shared_from_this<ZoneNode>
{
public:
    ZoneNode(ZoneId id,
             ZoneType type,
             double area,
             double perimeter,
             cv::Point2d centroid,
             std::vector<std::vector<cv::Point2d>> contours,
             std::optional<AABB> bbox,
             std::string label);

    /* IZoneNode getters implementation */
    ZoneId                                   id()         const noexcept override { return id_; }
    ZoneType                                 type()       const noexcept override { return type_; }
    const std::vector<Passage>&              neighbours() const noexcept override { return passages_; }
    double                                   area()       const noexcept override { return area_; }
    double                                   perimeter()  const noexcept override { return perimeter_; }
    const cv::Point2d&                           centroid()   const noexcept override { return centroid_; }
    const std::vector<std::vector<cv::Point2d>>& contours()   const noexcept override { return contours_; }

    /* internal for ZoneGraph */
    void addPassage(const NodePtr& neighbour, double width_m);
    void removePassageTo(const NodePtr& neighbour, double width_m);

    /* меняем тип после классификации */
    void setType(ZoneType t) noexcept { type_ = t; }

    /* упаковываем все числовые признаки в одну структуру */
    ZoneFeatures features() const noexcept;
private:
    ZoneId                                   id_{0};        ///< unique node id
    ZoneType                                 type_{ZoneType{}}; ///< semantic type
    double                                   area_{0.0};    ///< area in m²
    double                                   perimeter_{0.0}; ///< perimeter length
    cv::Point2d                              centroid_{};   ///< centroid in map coordinates
    std::vector<std::vector<cv::Point2d>>    contours_;     ///< polygonal contours
    std::optional<AABB>                      bbox_;         ///< optional bounding box
    std::string                              label_;        ///< custom textual label
    std::vector<Passage>                     passages_;     ///< connections to neighbours
};

/* ---------- concrete graph ------------------------------------------------- */
/**
 * @brief Concrete graph implementation storing ZoneNode objects.
 */
class ZoneGraph final : public IZoneGraph
{
public:
    NodePtr addNode(ZoneId id,
                    ZoneType type,
                    double area,
                    double perimeter,
                    cv::Point2d centroid,
                    std::vector<std::vector<cv::Point2d>> contours,
                    std::optional<AABB> bbox = std::nullopt,
                    std::string label = {}) override;

    bool    removeNode(ZoneId id) override;

    bool    connectZones   (ZoneId a, ZoneId b, double width_m) override;
    bool    disconnectZones(ZoneId a, ZoneId b, double width_m) override;

    NodePtr                 getNode(ZoneId id) const override;
    std::vector<NodePtr>    allNodes()          const override;

private:
    /** Helper for comparing floating point numbers. */
    static bool equalDouble(double a, double b, double eps = 1e-5);

    /** Storage of all nodes by their id. */
    std::unordered_map<ZoneId, NodePtr>      nodes_;
};

} // namespace mapping
