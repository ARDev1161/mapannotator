#pragma once
#include <string>
#include <unordered_map>
#include <sstream>
#include "../mapgraph/zonegraph.hpp"     // ваш граф

/* -------------------------------------------------------------- *
 * PDDLGenerator: формирует куски PDDL из ZoneGraph               *
 * -------------------------------------------------------------- */
/**
 * @brief Convert a ZoneGraph to fragments of PDDL domain/problem files.
 */
class PDDLGenerator
{
public:
    explicit PDDLGenerator(const mapping::ZoneGraph& g) : graph_(g) {}

    /* ---------- problem file helpers --------------------------- */
    /** PDDL (:objects ...) block. */
    std::string objects()             const;
    /** PDDL (:init ...) block. */
    std::string init(const std::string& start) const;
    /** PDDL (:goal ...) block. */
    std::string goal(const std::string& goal)  const;

    /* более мелкие под-функции, если нужно комбинировать вручную */
    /** Produce (class zoneX) predicates for all nodes. */
    std::string classPredicates()     const;

    /* ---------------------------------------------------------------
     *  Генерирует блок  (connected A B)
     *  sym == false  →   одна строка на ребро        A B
     *  sym == true   →   две строки на ребро         A B  и  B A
     *  Дубликаты исключаются независимо от того, хранит ли граф
     *  соседей с обеих сторон.
     * --------------------------------------------------------------- */
    /** Connectivity predicates optionally symmetric. */
    std::string connectivity(bool sym=false) const;

    /** (at robot Z) predicate helper. */
    std::string location(const std::string& z) const;

    /* ---------- helpers ---------------------------------------- */
    /** Sanitize zone label for use in PDDL identifiers. */
    std::string zoneLabel(const mapping::NodePtr& n) const;

private:
    const mapping::ZoneGraph&  graph_;
};
