#pragma once
#include <string>
#include <unordered_map>
#include <sstream>
#include "../mapgraph/zonegraph.hpp"     // ваш граф

/* -------------------------------------------------------------- *
 * PDDLGenerator: формирует куски PDDL из ZoneGraph               *
 * -------------------------------------------------------------- */
class PDDLGenerator
{
public:
    explicit PDDLGenerator(const mapping::ZoneGraph& g) : graph_(g) {}

    /* ---------- problem-файл ----------------------------------- */
    std::string objects()             const;          // (:objects …)
    std::string init(const std::string& start) const; // (:init …)
    std::string goal(const std::string& goal)  const; // (:goal …)

    /* более мелкие под-функции, если нужно комбинировать вручную */
    std::string classPredicates()     const;          // passage/room/…

    /* ---------------------------------------------------------------
     *  Генерирует блок  (connected A B)
     *  sym == false  →   одна строка на ребро        A B
     *  sym == true   →   две строки на ребро         A B  и  B A
     *  Дубликаты исключаются независимо от того, хранит ли граф
     *  соседей с обеих сторон.
     * --------------------------------------------------------------- */
    std::string connectivity(bool sym=false) const;    // connected

    std::string location(const std::string& z) const; // (at robot Z)

    /* ---------- helpers ---------------------------------------- */
    std::string zoneLabel(const mapping::NodePtr& n) const;    // нормализует имя

private:
    const mapping::ZoneGraph&  graph_;
};
