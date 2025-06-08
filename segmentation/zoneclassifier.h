#pragma once
#include <yaml-cpp/yaml.h>
#include <unordered_map>
#include <fstream>
#include <filesystem>
#include "../thirdparty/exprtk.hpp"
#include "../mapgraph/zoneconfig.hpp"

/**
 * @brief Rule-based classifier assigning semantic types to zones.
 */
class ZoneClassifier
{
    struct Rule
    {
        std::string       name;     ///< rule identifier
        const TypeInfo*   type{nullptr}; ///< resulting zone type
        int               priority; ///< higher value processed first
        exprtk::expression<double> expr; ///< compiled expression
        std::vector<std::string>   vars; ///< referenced feature/var names
    };

public:
    /** Construct classifier from a YAML rule file. */
    explicit ZoneClassifier(const std::string& yaml_file);

    /** Classify one zone, given its pre-computed geometric features. */
    ZoneType classify(const ZoneFeatures& z);

private:
    //------------------------------------------------------------------
    /** Parse YAML file and populate internal rule list. */
    void loadRules(const std::string& path);

    /** Convert YAML node into a TypeInfo pointer. */
    ZoneType decodeType(const YAML::Node& t);

    //------------------------------------------------------------------
    // Helpers
    double *addVar(const std::string& name);

//    static ZoneClass    str2class(const std::string&);
//    static ZoneSubClass str2sub(const std::string&);

    //------------------------------------------------------------------
    // Data members
    exprtk::symbol_table<double> syms_;
    exprtk::parser<double>       parser_;

    // Pointers to feature vars (for speed)
    double *A_, *P_, *C_, *AR_, *N_, *w_avg_, *w_min_;

    std::unique_ptr<TypeRegistry>      registry_;       ///< registry of zone types

    std::vector<Rule>                  rules_;          ///< ordered list of rules
    std::unordered_map<std::string,double*> rule_results_; ///< per-rule result flags
    std::unordered_map<std::string,double>  var_pool_;   ///< backing storage for variables
};
