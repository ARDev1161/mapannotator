#include "pddlgenerator.h"
#include <algorithm>
#include <unordered_set>
#include <cctype>

using namespace std;

/* маленький утилити-help */
static string toLower(string s){
    transform(s.begin(), s.end(), s.begin(),
              [](unsigned char c){ return std::tolower(c); });
    return s;
}

/* 1. имена объектов (robot + все зоны) ------------------------- */
string PDDLGenerator::objects() const
{
    ostringstream oss;
    oss << "  (:objects\n"
        << "    robot               - robot\n";
    for (const auto& n : graph_.allNodes())
        oss << "    " << zoneLabel(n) << "   - zone\n";
    oss << "  )\n";
    return oss.str();
}

/* 2. предикаты-классы зон -------------------------------------- */
string PDDLGenerator::classPredicates() const
{
    ostringstream oss;
    for (const auto& n : graph_.allNodes())
        oss << "    (" << toLower(n->type().major())
            << ' '   << zoneLabel(n) << ")\n";
    return oss.str();
}

/* 3. connected A B (симметрично) ------------------------------- */
string PDDLGenerator::connectivity(bool sym) const
{
    using ZoneId  = mapping::ZoneId;
    using EdgeKey = std::pair<ZoneId, ZoneId>;          // упорядоченная пара id

    auto make_key = [](ZoneId a, ZoneId b) -> EdgeKey   // undirected
    { return (a < b) ? EdgeKey{a, b} : EdgeKey{b, a}; };

    struct EdgeHash
    {
        std::size_t operator()(const EdgeKey& k) const noexcept
        {
            return std::hash<ZoneId>{}(k.first) ^
                   (std::hash<ZoneId>{}(k.second) << 1);
        }
    };

    std::unordered_set<EdgeKey, EdgeHash> printed;      // уже выведенные рёбра
    std::ostringstream                    oss;

    for (const auto& n : graph_.allNodes())
    {
        const ZoneId idA = n->id();
        const std::string labelA = zoneLabel(n);

        for (const auto& p : n->neighbours())
        {
            auto nb = p.neighbour.lock();
            if (!nb) continue;

            const ZoneId idB = nb->id();
            EdgeKey key      = make_key(idA, idB);

            /* Печатаем ребро только при первом появлении этой пары id */
            if (!printed.insert(key).second) continue;  // уже было

            const std::string labelB = zoneLabel(nb);

            oss << "    (connected " << labelA << ' ' << labelB << ")\n";
            if (sym)
                oss << "    (connected " << labelB << ' ' << labelA << ")\n";
        }
    }
    return oss.str();
}

/* 4. (at robot Z) ---------------------------------------------- */
string PDDLGenerator::location(const string& z) const
{
    ostringstream oss;
    oss << "    (at robot " << z << ")\n";
    return oss.str();
}

/* 5. Полный (:init …) ------------------------------------------ */
string PDDLGenerator::init(const string& start) const
{
    ostringstream oss;
    oss << "  (:init\n";
    oss << classPredicates();
    oss << connectivity();
    oss << location(start);
    oss << "  )\n";
    return oss.str();
}

/* 6. Goal ------------------------------------------------------- */
string PDDLGenerator::goal(const string& goal) const
{
    ostringstream oss;
    oss << "  (:goal (and (at robot " << goal << ") ))\n";
    return oss.str();
}

/* 7. label helper ---------------------------------------------- */
string PDDLGenerator::zoneLabel(const mapping::NodePtr& n) const
{
    /* 1. берём строку "passage.doorarea" … */
    std::string label = n->type().info
                        ? n->type().info->path        // "passage.doorarea"
                        : "unknown";

    /* 2. заменяем все точки на подчёркивание */
    std::replace(label.begin(), label.end(), '.', '_'); // <algorithm>

    /* 3. добавляем числовой id узла */
    label += '_' + std::to_string(n->id());             // → "passage_doorarea_4"

    /* 4. приводим к нижнему регистру (если нужно) */
    return toLower(label);
}
