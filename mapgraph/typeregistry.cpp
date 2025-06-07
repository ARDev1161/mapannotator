#include "typeregistry.h"
#include <stdexcept>

void TypeRegistry::parseNode(const YAML::Node& node,
                             const std::string& parent_path)
{
    for (const auto& kv : node)
    {
        const std::string name = kv.first.as<std::string>();
        const YAML::Node body = kv.second;

        const std::string path = parent_path.empty()
                               ? name
                               : parent_path + "." + name;

        const uint16_t id = body["id"]
                          ? body["id"].as<uint16_t>()
                          : static_cast<uint16_t>(by_path_.size() + 1);

        /* 1. создаём объект и заполняем все поля ПЕРЕД перемещением */
        auto ti  = std::make_unique<TypeInfo>();
        ti->id   = id;
        ti->name = name;
        ti->path = path;
        ti->major = parent_path.empty()
                  ? name
                  : parent_path.substr(0, parent_path.find('.'));

        if (body["color"])
        {
            const auto& v = body["color"];      // [B,G,R]
            ti->color = cv::Scalar(v[0].as<int>(),
                                   v[1].as<int>(),
                                   v[2].as<int>());
        }

        /* raw-указатель нужен и для unknown_, и для возможных ссылок */
        TypeInfo* ti_raw = ti.get();

        /* 2. кладём в map (после всех инициализаций) */
        by_path_[path] = std::move(ti);

        /* 3. special-case "Unknown" */
        if (name == "Unknown")
            unknown_ = ti_raw;

        /* 4. рекурсия по children */
        if (body["children"])
            parseNode(body["children"], path);
    }
}

bool TypeRegistry::load(const YAML::Node &types_y)
{
    by_path_.clear();
    unknown_ = nullptr;

    if (!types_y || !types_y.IsMap())
        throw std::runtime_error("TypeRegistry::load(): 'types' node not a map");

    /* пройти все корневые элементы */
    parseNode(types_y, /*parent_path=*/"");

    /* гарантируем, что Unknown есть в карте,
       иначе указываем на фиктивный nullptr и вернём false */
    return (unknown_ != nullptr);
}

const TypeInfo* TypeRegistry::get(const std::string& p) const
{
    auto it = by_path_.find(p);
    return it==by_path_.end() ? unknown_ : it->second.get();
}
