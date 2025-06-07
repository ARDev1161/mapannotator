#ifndef TYPEREGISTRY_H
#define TYPEREGISTRY_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <memory>
#include <optional>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

/* 2.1 Информация о типе, расширяемая без перекомпиляции */
struct TypeInfo
{
    uint16_t            id;         // уникальный числовой код
    std::string         name;       // "Corridor"
    std::string         path;       // "Passage.Corridor"
    std::string         major;      // "Passage"
    std::optional<cv::Scalar> color;   // B,G,R, optionally loaded
};

/* 2.2 Итоговый тип зоны, который возвращает классификатор */
struct ZoneType
{
    /*  pointer to the entry in TypeRegistry;
        nullptr  ⇒  «Unknown / not classified»                         */
    std::shared_ptr<const TypeInfo> info;              // ← имеет дефолтное значение

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

inline cv::Scalar zoneColor(const ZoneType& z)
{
    /* 1. YAML-цвет задан → используем его */
    if (z.info && z.info->color) return *z.info->color;

    /* 2. иначе  – автоматически: hash(path) → пастельный цвет */
    std::size_t h = std::hash<std::string>{}( z.path() );
    int b = 150 + (h & 0x3F);        // 150-213  (пастель)
    int g = 150 + ((h >> 6 ) & 0x3F);
    int r = 150 + ((h >> 12) & 0x3F);
    return cv::Scalar(b,g,r);
}

/*  ✧✧✧  хеш-функция для unordered_map  ✧✧✧  */
namespace std
{
    template<> struct hash<ZoneType>
    {
        size_t operator()(const ZoneType& z) const noexcept
        {
            return std::hash<std::shared_ptr<const TypeInfo>>{}(z.info);
        }
    };
}

class TypeRegistry
{
public:
    bool load(const YAML::Node& types_y);              // рекурсивный разбор
    const TypeInfo* get(const std::string& path) const; // "A.B"
    const TypeInfo* unknown() const { return unknown_; }

private:
    std::unordered_map<std::string, std::unique_ptr<TypeInfo>> by_path_;
    const TypeInfo* unknown_{nullptr};

    void parseNode(const YAML::Node& node,
                   const std::string& parent_path);
};

#endif // TYPEREGISTRY_H
