#pragma once
#include <yaml-cpp/yaml.h>
#include <unordered_map>
#include <fstream>
#include <filesystem>
#include "../thirdparty/exprtk.hpp"
#include "../mapgraph/zoneconfig.hpp"

class ZoneClassifier
{
    struct Rule
    {
        std::string       name;
        const TypeInfo*                type{nullptr};
        int               priority;
        exprtk::expression<double> expr; // compiled
        std::vector<std::string>   vars; // feature + rule vars referenced
    };

public:
    explicit ZoneClassifier(const std::string& yaml_file);

    /** Classify one zone, given its pre-computed geometric features. */
    ZoneType classify(const ZoneFeatures& z);

private:
    //------------------------------------------------------------------
    // YAML → internal rule list
    void loadRules(const std::string& path);

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

    std::unique_ptr<TypeRegistry>      registry_;

    std::vector<Rule>                         rules_;
    std::unordered_map<std::string,double*>   rule_results_;
    std::unordered_map<std::string,double>    var_pool_; // backing storage
};
