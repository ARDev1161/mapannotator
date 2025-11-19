#include "downsample_seeding.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

namespace
{
struct SeedLayer
{
    cv::Mat1i labels;   ///< connected components of free cores
};

using SeedMaskMap = std::unordered_map<int, cv::Mat1b>;

/** Extract unique positive labels from CV_32S map. */
static std::vector<int> uniqueLabels(const cv::Mat1i &labels)
{
    CV_Assert(labels.type() == CV_32S);
    std::set<int> uniq;
    for (int y = 0; y < labels.rows; ++y) {
        const int *row = labels.ptr<int>(y);
        for (int x = 0; x < labels.cols; ++x) {
            int v = row[x];
            if (v > 0)
                uniq.insert(v);
        }
    }
    return std::vector<int>(uniq.begin(), uniq.end());
}

/** Helper that materialises a binary mask for the requested label. */
static cv::Mat1b labelMask(const cv::Mat1i &labels, int label)
{
    cv::Mat1b mask(labels.size(), uchar(0));
    cv::compare(labels, label, mask, cv::CMP_EQ);
    return mask;
}

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
    cv::Mat1f freeNorm;
    srcBinary.convertTo(freeNorm, CV_32F, 1.0 / 255.0);
    cv::Mat1b backgroundMask = buildBackgroundMask(srcBinary,
                                                   cfg.backgroundKernel);

    std::vector<SeedLayer> layers;
    double sigma = cfg.sigmaStart;
    for (int iter = 0; iter < cfg.maxIter; ++iter) {
        cv::Mat blurred;
        cv::GaussianBlur(freeNorm, blurred, cv::Size(), sigma, sigma,
                         cv::BORDER_REPLICATE);
        cv::Mat thresholded;
        cv::threshold(blurred, thresholded, cfg.threshold, 1.0,
                      cv::THRESH_BINARY);
        cv::Mat1b bin;
        thresholded.convertTo(bin, CV_8U, 255);
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

/** Register or update the seed mask associated with a global id. */
static void registerSeed(SeedMaskMap &seedMasks,
                         int globalId,
                         const cv::Mat1b &mask)
{
    seedMasks[globalId] = mask.clone();
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

    SeedMaskMap seedMasks;
    int nextGlobalId = 1;
    std::unordered_map<int, int> prevLocalToGlobal;

    for (int layerIdx = static_cast<int>(layers.size()) - 1;
         layerIdx >= 0;
         --layerIdx)
    {
        const auto &layer = layers[layerIdx];
        std::vector<int> currentLabels = uniqueLabels(layer.labels);
        std::unordered_map<int, int> currentLocalToGlobal;
        std::unordered_map<int, int> parentChildCount;

        struct LabelParents {
            int label;
            std::vector<int> parents;
            cv::Mat1b mask;
        };
        std::vector<LabelParents> info;
        info.reserve(currentLabels.size());

        if (layerIdx == static_cast<int>(layers.size()) - 1) {
            for (int lbl : currentLabels) {
                cv::Mat1b mask = labelMask(layer.labels, lbl);
                registerSeed(seedMasks, nextGlobalId, mask);
                currentLocalToGlobal[lbl] = nextGlobalId++;
            }
        } else {
            const auto &prevLayer = layers[layerIdx + 1];

            for (int lbl : currentLabels) {
                LabelParents entry;
                entry.label = lbl;
                entry.mask  = labelMask(layer.labels, lbl);

                std::set<int> parents;
                for (int y = 0; y < entry.mask.rows; ++y) {
                    const uchar *rowMask = entry.mask.ptr<uchar>(y);
                    const int   *rowPrev = prevLayer.labels.ptr<int>(y);
                    for (int x = 0; x < entry.mask.cols; ++x) {
                        if (rowMask[x]) {
                            int prevLbl = rowPrev[x];
                            if (prevLbl > 0)
                                parents.insert(prevLbl);
                        }
                    }
                }
                entry.parents.assign(parents.begin(), parents.end());
                for (int parentLocal : entry.parents)
                    parentChildCount[parentLocal]++;
                info.push_back(std::move(entry));
            }

            for (const auto &entry : info) {
                if (entry.parents.empty()) {
                    int gid = nextGlobalId++;
                    registerSeed(seedMasks, gid, entry.mask);
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
                        registerSeed(seedMasks, gid, entry.mask);
                        currentLocalToGlobal[entry.label] = gid;
                        continue;
                    }

                    if (parentChildCount[parentLocal] == 1) {
                        currentLocalToGlobal[entry.label] = parentGlobal;
                        registerSeed(seedMasks, parentGlobal, entry.mask);
                    } else {
                        seedMasks.erase(parentGlobal); // родитель не лист
                        int gid = nextGlobalId++;
                        registerSeed(seedMasks, gid, entry.mask);
                        currentLocalToGlobal[entry.label] = gid;
                    }
                } else {
                    int gid = nextGlobalId++;
                    registerSeed(seedMasks, gid, entry.mask);
                    currentLocalToGlobal[entry.label] = gid;
                }
            }
        }

        prevLocalToGlobal = std::move(currentLocalToGlobal);
    }

    std::vector<int> ids;
    ids.reserve(seedMasks.size());
    for (const auto &kv : seedMasks)
        ids.push_back(kv.first);
    std::sort(ids.begin(), ids.end());

    std::vector<ZoneMask> seeds;
    seeds.reserve(ids.size());
    for (int id : ids) {
        seeds.push_back({id, seedMasks[id]});
    }
    return seeds;
}
