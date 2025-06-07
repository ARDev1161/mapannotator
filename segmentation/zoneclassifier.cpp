#include "zoneclassifier.h"

ZoneClassifier::ZoneClassifier(const std::string &yaml_file)
{
    loadRules(yaml_file);
}

ZoneType ZoneClassifier::classify(const ZoneFeatures &z)
{
    /* 1. заполняем числовые признаки */
    *A_      = z.A;
    *P_      = z.P;
    *C_      = z.C;
    *AR_     = z.AR;
    *N_      = static_cast<double>(z.N);
    *w_avg_  = z.w_avg;
    *w_min_  = z.w_min;

    /* 2. Обнуляем булевы переменные, представляющие имена правил,
          — чтобы результаты предыдущей зоны не «протекали» */
    for (auto& kv : rule_results_)
        *kv.second = 0.0;           // 0 = false

    /* 3. Идём по правилам в порядке приоритета */
    for (const auto& rule : rules_)
    {
        if (rule.expr.value())      // ExprTk уже знает все переменные
        {
            *rule_results_[rule.name] = 1.0;   // этот rule стал true
            return ZoneType{ rule.type };                  // ← нашли нужный ZoneType
        }
    }

    /* 4. Fallback */
    return ZoneType{ registry_->unknown() };
}

void ZoneClassifier::loadRules(const std::string &path)
{
    YAML::Node root = YAML::LoadFile(path);

    /* 0. построить реестр типов */
    registry_ = std::make_unique<TypeRegistry>();
    if (!root["types"] || !registry_->load(root["types"]))
        throw std::runtime_error("ZoneClassifier: section 'types' missing or invalid");

    const YAML::Node yaml_rules = root["rules"];
    if (!yaml_rules)
        throw std::runtime_error("ZoneClassifier: section 'rules' missing");

    // Common symbol table for ExprTk(shared among all rules)
    syms_.add_constants();

    // ❶ Feature variables
    A_     = addVar("A");
    P_     = addVar("P");
    C_     = addVar("C");
    AR_    = addVar("AR");
    N_     = addVar("N");
    w_avg_ = addVar("w_avg");
    w_min_ = addVar("w_min");

    // ❷ For rule dependency tracking we create one double per rule name
    for (const auto& y : yaml_rules)
    {
        const std::string name = y["name"].as<std::string>();
        rule_results_[name] = addVar(name);   // default 0
    }

    // ❸ Parse every rule, compile expression
    for (const auto& y : yaml_rules)
    {
        Rule r;
        r.name     = y["name"].as<std::string>();
        r.priority = y["priority"] ? y["priority"].as<int>() : 0;

        /* путь типа, например "Passage.Corridor" */
        std::string type_path = y["type"].as<std::string>();
        r.type = registry_->get(type_path);
        if (!r.type)
            throw std::runtime_error("Unknown type path '" + type_path +
                                     "' used in rule '" + r.name + "'");

        /* компиляция выражения */
        r.expr.register_symbol_table(syms_);
        const std::string expr_str = y["expr"].as<std::string>();
        if (!parser_.compile(expr_str, r.expr))
            throw std::runtime_error("ExprTk error in rule '" + r.name +
                                     "': " + parser_.error());

        rules_.push_back(std::move(r));
    }

    // ❹ Sort by priority ↓
    std::sort(rules_.begin(), rules_.end(),
              [](const Rule& a, const Rule& b){ return a.priority > b.priority; });
}

double *ZoneClassifier::addVar(const std::string &name)
{
    syms_.add_variable(name, var_pool_[name]); // var_pool_ default-initialised to zero
    return &var_pool_[name];
}
