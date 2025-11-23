#include "downsample_seeding.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <limits>
#include <iostream>

namespace
{
struct SeedLayer
{
    cv::Mat1i labels;   ///< connected components of free cores
};

/** Build a background mask using a small erosion kernel. */
static cv::Mat1b buildBackgroundMask(const cv::Mat1b &srcBinary,
                                     int kernelSize)
{
    CV_Assert(srcBinary.type() == CV_8UC1);
    int k = std::max(1, kernelSize);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                               cv::Size(k, k));
    cv::Mat eroded;
    cv::erode(srcBinary, eroded, kernel);
    return (eroded == 0);
}

/** Generate all seed layers by iteratively smoothing the free space map. */
static std::vector<SeedLayer>
buildSeedLayers(const cv::Mat1b &srcBinary, const DownsampleSeedsConfig &cfg)
{
    cv::Mat1b backgroundMask = buildBackgroundMask(srcBinary,
                                                   cfg.backgroundKernel);

    std::vector<SeedLayer> layers;
    double sigma = cfg.sigmaStart;
    for (int iter = 0; iter < cfg.maxIter; ++iter) {
        cv::Mat blurred;
        cv::GaussianBlur(srcBinary, blurred, cv::Size(), sigma, sigma,
                         cv::BORDER_REPLICATE);
        cv::Mat1b bin;
        const double thresh = std::clamp(cfg.threshold, 0.0, 1.0) * 255.0;
        cv::threshold(blurred, bin, thresh, 255.0, cv::THRESH_BINARY);
        bin.setTo(0, backgroundMask);               // уверенный фон = препятствие

        if (cv::countNonZero(bin) == 0)
            break;

        cv::Mat1i labels;
        int nLabels = cv::connectedComponents(bin, labels, 8, CV_32S);
        if (nLabels <= 1)
            break;

        layers.push_back({labels});
        sigma += cfg.sigmaStep;
    }
    return layers;
}

} // namespace

namespace {
struct LabelInfo
{
    int label = 0;
    int area = 0;
    int minX = std::numeric_limits<int>::max();
    int minY = std::numeric_limits<int>::max();
    int maxX = std::numeric_limits<int>::min();
    int maxY = std::numeric_limits<int>::min();
    std::vector<int> parents;

    void updateBBox(int x, int y)
    {
        minX = std::min(minX, x);
        minY = std::min(minY, y);
        maxX = std::max(maxX, x);
        maxY = std::max(maxY, y);
    }
};

static std::vector<LabelInfo>
collectLabelInfo(const cv::Mat1i &labels, const cv::Mat1i *prevLayer)
{
    std::unordered_map<int, LabelInfo> info;
    info.reserve(256);

    for (int y = 0; y < labels.rows; ++y)
    {
        const int *rowCurr = labels.ptr<int>(y);
        const int *rowPrev = prevLayer ? prevLayer->ptr<int>(y) : nullptr;
        for (int x = 0; x < labels.cols; ++x)
        {
            int lbl = rowCurr[x];
            if (lbl <= 0)
                continue;

            auto &entry = info[lbl];
            if (entry.label == 0)
                entry.label = lbl;

            entry.area++;
            entry.updateBBox(x, y);

            if (rowPrev)
            {
                int parent = rowPrev[x];
                if (parent > 0 &&
                    std::find(entry.parents.begin(), entry.parents.end(), parent) == entry.parents.end())
                {
                    entry.parents.push_back(parent);
                }
            }
        }
    }

    std::vector<LabelInfo> result;
    result.reserve(info.size());
    for (auto &kv : info)
        result.push_back(std::move(kv.second));

    std::sort(result.begin(), result.end(),
              [](const LabelInfo &a, const LabelInfo &b) { return a.label < b.label; });
    return result;
}
} // namespace

std::vector<ZoneMask>
generateDownsampleSeeds(const cv::Mat1b &srcBinary,
                        const DownsampleSeedsConfig &cfg)
{
    CV_Assert(srcBinary.type() == CV_8UC1);

    auto layers = buildSeedLayers(srcBinary, cfg);
    if (layers.empty())
        return {};

    cv::Mat1i seedLabelMap(srcBinary.size(), int(0)); // global seed id per pixel
    std::unordered_set<int> activeSeeds;
    std::vector<int> seedOrder;
    seedOrder.reserve(cfg.maxSeeds > 0 ? cfg.maxSeeds : 256);
    cv::Mat1b scratchMask(srcBinary.size(), uchar(0)); // reused buffer
    int nextGlobalId = 1;
    std::unordered_map<int, int> prevLocalToGlobal;
    bool abortedByLimit = false;

    auto tryRegisterSeed = [&](int globalId,
                               const cv::Mat1i &labelMap,
                               const LabelInfo &info) {
        if (cfg.minSeedAreaPx > 0 && info.area < cfg.minSeedAreaPx)
            return true; // skip tiny seeds silently

        if (cfg.maxSeeds > 0 &&
            static_cast<int>(activeSeeds.size()) >= cfg.maxSeeds)
        {
            abortedByLimit = true;
            return false;
        }

        cv::compare(labelMap, info.label, scratchMask, cv::CMP_EQ);
        seedLabelMap.setTo(globalId, scratchMask);

        if (activeSeeds.insert(globalId).second)
            seedOrder.push_back(globalId);
        return true;
    };

    for (int layerIdx = static_cast<int>(layers.size()) - 1;
         layerIdx >= 0;
         --layerIdx)
    {
        const auto &layer = layers[layerIdx];
        const cv::Mat1i *prevLabels = (layerIdx + 1 < static_cast<int>(layers.size()))
                                      ? &layers[layerIdx + 1].labels
                                      : nullptr;

        std::vector<LabelInfo> infos = collectLabelInfo(layer.labels, prevLabels);
        std::unordered_map<int, int> currentLocalToGlobal;
        std::unordered_map<int, int> parentChildCount;

        for (const auto &entry : infos)
        {
            for (int parentLocal : entry.parents)
                parentChildCount[parentLocal]++;
        }

        if (!prevLabels) {
            for (const auto &entry : infos) {
                int gid = nextGlobalId++;
                if (!tryRegisterSeed(gid, layer.labels, entry))
                    break;
                currentLocalToGlobal[entry.label] = gid;
            }
        } else {
            for (const auto &entry : infos) {
                if (abortedByLimit)
                    break;

                if (entry.parents.empty()) {
                    int gid = nextGlobalId++;
                    if (!tryRegisterSeed(gid, layer.labels, entry))
                        break;
                    currentLocalToGlobal[entry.label] = gid;
                    continue;
                }

                if (entry.parents.size() == 1) {
                    int parentLocal  = entry.parents.front();
                    auto parentIt    = prevLocalToGlobal.find(parentLocal);
                    int parentGlobal = parentIt != prevLocalToGlobal.end()
                                       ? parentIt->second : -1;
                    if (parentGlobal < 0) {
                        int gid = nextGlobalId++;
                        if (!tryRegisterSeed(gid, layer.labels, entry))
                            break;
                        currentLocalToGlobal[entry.label] = gid;
                        continue;
                    }

                    if (parentChildCount[parentLocal] == 1) {
                        currentLocalToGlobal[entry.label] = parentGlobal;
                        if (!tryRegisterSeed(parentGlobal, layer.labels, entry))
                            break;
                    } else {
                        if (activeSeeds.erase(parentGlobal))
                        {
                            cv::compare(seedLabelMap, parentGlobal, scratchMask, cv::CMP_EQ);
                            seedLabelMap.setTo(0, scratchMask);
                        }
                        int gid = nextGlobalId++;
                        if (!tryRegisterSeed(gid, layer.labels, entry))
                            break;
                        currentLocalToGlobal[entry.label] = gid;
                    }
                } else {
                    int gid = nextGlobalId++;
                    if (!tryRegisterSeed(gid, layer.labels, entry))
                        break;
                    currentLocalToGlobal[entry.label] = gid;
                }
            }
        }

        if (abortedByLimit)
            break;

        prevLocalToGlobal = std::move(currentLocalToGlobal);
    }

    if (abortedByLimit) {
        std::cerr << "[warn] down-sample seeding aborted: too many seeds "
                  << "(cap=" << cfg.maxSeeds << ")\n";
        return {};
    }

    std::vector<ZoneMask> seeds;
    seeds.reserve(seedOrder.size());
    for (int id : seedOrder) {
        if (activeSeeds.find(id) == activeSeeds.end())
            continue;
        cv::Mat1b mask;
        cv::compare(seedLabelMap, id, mask, cv::CMP_EQ);
        seeds.push_back({id, std::move(mask)});
    }
    return seeds;
}
