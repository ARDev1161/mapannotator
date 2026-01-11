#ifndef TYPEREGISTRY_H
#define TYPEREGISTRY_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <memory>
#include <optional>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

/**
 * @brief Runtime description of a semantic zone type.
 *
 * Instances of this struct are loaded from YAML and stored inside the
 * TypeRegistry. They are referenced by ZoneType objects via shared pointers.
 */
struct TypeInfo
{
    uint16_t            id;    ///< unique numeric identifier
    std::string         name;  ///< short name, e.g. "Corridor"
    std::string         path;  ///< full path like "Passage.Corridor"
    std::string         major; ///< top level category
    std::optional<cv::Scalar> color; ///< optional BGR colour
};

/**
 * @brief Final zone type returned by the classifier.
 *
 * The class merely wraps a pointer to TypeInfo. If the pointer is null the type
 * is considered unknown.
 */
struct ZoneType
{
    /** Pointer to the registry entry; not owned. May be nullptr for "Unknown". */
    const TypeInfo* info{nullptr};

    /* convenience ctors */
    ZoneType()  = default;
    explicit ZoneType(const TypeInfo* i) : info{i} {}

    /* comparisons */
    bool operator==(const ZoneType& o) const noexcept { return info == o.info; }
    bool operator!=(const ZoneType& o) const noexcept { return info != o.info; }

    std::string path()  const { return info ? info->path  : "Unknown"; }
    uint16_t    id()    const { return info ? info->id    : 0; }
    std::string major() const { return info ? info->major : "Unknown"; }
};

/**
 * @brief Pick a display colour for the given zone type.
 *
 * If the type has an explicit colour in the YAML description it is used;
 * otherwise a pastel shade derived from the type path is returned.
 */
inline cv::Scalar zoneColor(const ZoneType& z)
{
    /* 1. YAML colour specified → use it */
    if (z.info && z.info->color) return *z.info->color;

    /* 2. otherwise derive a stronger colour from hash(path) */
    std::size_t h = std::hash<std::string>{}( z.path() );
    int b = 60 + (h & 0x8F);        // 60-203
    int g = 60 + ((h >> 7 ) & 0x8F);
    int r = 60 + ((h >> 14) & 0x8F);
    return cv::Scalar(b,g,r);
}

/*  ✧✧✧  хеш-функция для unordered_map  ✧✧✧  */
namespace std
{
    template<> struct hash<ZoneType>
    {
        size_t operator()(const ZoneType& z) const noexcept
        {
            return std::hash<const TypeInfo*>{}(z.info);
        }
    };
}

/**
 * @brief Container storing all known zone types loaded from YAML.
 */
class TypeRegistry
{
public:
    /** Parse YAML tree and populate the registry recursively. */
    bool load(const YAML::Node& types_y);

    /** Retrieve type information by path, e.g. "A.B". */
    const TypeInfo* get(const std::string& path) const;

    /** Pointer to the predefined "Unknown" type (may be nullptr). */
    const TypeInfo* unknown() const { return unknown_; }

private:
    /** Map path string → TypeInfo instance. */
    std::unordered_map<std::string, std::unique_ptr<TypeInfo>> by_path_;

    /** Cached pointer to the "Unknown" entry. */
    const TypeInfo* unknown_{nullptr};

    /** Recursive helper used by load(). */
    void parseNode(const YAML::Node& node,
                   const std::string& parent_path);
};

#endif // TYPEREGISTRY_H
