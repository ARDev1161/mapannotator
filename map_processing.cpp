#include "map_processing.hpp"

#include <fstream>
#include <unordered_set>
#include <opencv2/ximgproc.hpp>

#include "mapgraph/zone_graph_dot.hpp"
#include "mapgraph/zone_graph_draw.hpp"
#include "segmentation/zoneclassifier.h"
#include "utils.hpp"

using namespace mapping;

//--------------------------------------------------------------------------
//  buildZoneAdjacency()
//--------------------------------------------------------------------------
static std::unordered_map<int, std::vector<int>>
buildZoneAdjacency(const cv::Mat1i &zones, bool use8Connectivity = false)
{
    CV_Assert(!zones.empty() && zones.type() == CV_32S);

    const cv::Size sz = zones.size();

    const std::vector<cv::Point> off4{{1,0},{0,1}};
    const std::vector<cv::Point> off8{{1,0},{0,1},{1,1},{1,-1}};
    const auto &offs = use8Connectivity ? off8 : off4;

    std::unordered_map<int, std::unordered_set<int>> tmpAdj;

    for (int y = 0; y < sz.height; ++y)
    {
        const int *row = zones.ptr<int>(y);
        for (int x = 0; x < sz.width; ++x)
        {
            int lbl = row[x];
            if (lbl <= 0) continue;                     // фон

            for (const auto &d : offs)
            {
                int nx = x + d.x, ny = y + d.y;
                if ((unsigned)nx >= (unsigned)sz.width ||
                    (unsigned)ny >= (unsigned)sz.height) continue;

                int nLbl = zones.at<int>(ny, nx);
                if (nLbl <= 0 || nLbl == lbl) continue; // тот же или фон

                tmpAdj[lbl].insert(nLbl);
                tmpAdj[nLbl].insert(lbl);               // граф неориентированный
            }
        }
    }

    std::unordered_map<int, std::vector<int>> adjacency;
    adjacency.reserve(tmpAdj.size());

    for (auto &[lbl, set] : tmpAdj)
    {
        adjacency[lbl] = {set.begin(), set.end()};
        std::sort(adjacency[lbl].begin(), adjacency[lbl].end());
    }
    return adjacency;
}

//--------------------------------------------------------------------------
//  buildZoneIndex()
//--------------------------------------------------------------------------
static std::unordered_map<int, const cv::Mat1b *>
buildZoneIndex(const std::vector<ZoneMask> &zones)
{
    std::unordered_map<int, const cv::Mat1b *> index;
    index.reserve(zones.size());

    for (const auto &z : zones)
        index.emplace(z.label, &z.mask);

    return index;
}

//--------------------------------------------------------------------------
//  annotateGraph()
//--------------------------------------------------------------------------
static void annotateGraph(ZoneGraph &graph, const std::string &rulefile)
{
    static ZoneClassifier clf{rulefile};

    for (auto &node : graph.allNodes())
    {
        ZoneFeatures f = node->features();
        ZoneType     t = clf.classify(f);

        node->setType(t);
        std::cerr << "Zone #" << node->id()
                  << " → " << t.path() << '\n';
    }
}

//--------------------------------------------------------------------------
//  buildGraph()
//--------------------------------------------------------------------------
void buildGraph(ZoneGraph &graphOut, std::vector<ZoneMask> zones,
                cv::Mat1i zonesMat,
                std::unordered_map<int, cv::Point> centroids)
{
    std::unordered_map<ZoneType, NodePtr> node;   // для удобства соединений

    auto add = [&](ZoneId id, ZoneType t, cv::Point2d c, double area,
                   double perimeter)
    {
        node[t] = graphOut.addNode(id, t, area, perimeter, c,
                                   {{{c.x-0.5,c.y-0.5},{c.x+0.5,c.y-0.5},
                                     {c.x+0.5,c.y+0.5},{c.x-0.5,c.y+0.5}}});
    };

    auto W = [](double m){ return m; };        // удобный alias

    std::unordered_map<int, std::vector<int>> zoneGraph =
            buildZoneAdjacency(zonesMat, /*use8Connectivity=*/true);

    // Assume map resolution is 5 cm per pixel ⇒ area per pixel = 0.05 × 0.05 m².
    constexpr double kAreaPerPixel = 0.05 * 0.05;

    auto zoneIndex = buildZoneIndex(zones);

    // Создаем ноды графа
    for (auto centroid : centroids)
    {
        // размещаем вершины (координаты в «метрах»)
        add(centroid.first,
            ZoneType{},
            pixelToWorld(centroid.second, mapInfo),
            computeWhiteArea(zoneIndex[centroid.first], kAreaPerPixel),
            10);

        // строим рёбра
        std::cerr << "Zone " << std::to_string(centroid.first) << centroid.second
                  << " neighbours:";
        for (auto neighbour : zoneGraph[centroid.first])
        {
            std::cerr << ' ' << neighbour;
            graphOut.connectZones(centroid.first, neighbour, W(1.4));
        }
        std::cerr << '\n';
    }

    annotateGraph(graphOut, "../mapannotator/config/rules.yaml");

    /* ---------- DOT‑файл ----------------------------------------------- */
    std::ofstream dot("graph.dot");
    writeDot(graphOut, dot);

    /* ---------- Быстрый предварительный просмотр (OpenCV) -------------- */
    cv::Mat img;
    drawZoneGraph(graphOut, img, 50.0 /*px/м*/, 7 /*радиус*/, true /*подписи ширины*/);
    showMat("Floor‑Plan Graph", img);
    cv::imwrite("graph_preview.png", img);
}

//--------------------------------------------------------------------------
//  buildFloodMask()
//--------------------------------------------------------------------------
static cv::Mat1b buildFloodMask(const cv::Mat1b &freeMap,
                                const cv::Point &seed,
                                int connectivity = 8)
{
    CV_Assert(freeMap.type() == CV_8UC1);
    CV_Assert(seed.x >= 0 && seed.x < freeMap.cols &&
              seed.y >= 0 && seed.y < freeMap.rows);

    if (freeMap(seed) != 0)
        return cv::Mat1b::zeros(freeMap.size());

    cv::Mat1b mask(freeMap.rows + 2, freeMap.cols + 2, uchar(0));

    const int flags = connectivity | cv::FLOODFILL_MASK_ONLY | (255 << 8);
    cv::floodFill(freeMap, mask, seed, 0,
                  nullptr,
                  cv::Scalar(), cv::Scalar(),
                  flags);

    return mask(cv::Rect(1, 1, freeMap.cols, freeMap.rows)).clone();
}

//--------------------------------------------------------------------------
//  mergeLonelyFreeAreas()
//--------------------------------------------------------------------------
static int mergeLonelyFreeAreas(cv::Mat1b &occ,
                                const cv::Mat1b &background,
                                std::vector<ZoneMask> &allZones)
{
    CV_Assert( occ.size() == background.size() &&
               occ.type() == CV_8UC1 && background.type() == CV_8UC1 );

    cv::Mat1i labelMap = cv::Mat1i::zeros(occ.size());
    std::unordered_map<int, std::size_t> label2idx;

    for (std::size_t i = 0; i < allZones.size(); ++i)
    {
        const auto &z = allZones[i];
        CV_Assert(z.mask.size() == occ.size() && z.mask.type() == CV_8UC1);
        labelMap.setTo(z.label, z.mask);
        label2idx[z.label] = i;
    }

    cv::Mat1b freeMask;
    cv::bitwise_and(occ == 0, background, freeMask);

    cv::Mat1i compLabels;
    int nComp = cv::connectedComponents(freeMask, compLabels, 8, CV_32S);

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,{3,3});
    int mergedCount = 0;

    for (int id = 1; id < nComp; ++id)
    {
        cv::Mat1b regionMask = (compLabels == id);

        cv::Mat1b dilated;
        cv::dilate(regionMask, dilated, kernel, {-1,-1}, 1);

        std::vector<cv::Point> pts;
        cv::findNonZero(dilated, pts);

        std::unordered_set<int> adjLabels;
        for (const auto &p : pts)
        {
            int lbl = labelMap.at<int>(p);
            if (lbl > 0) adjLabels.insert(lbl);
        }

        if (adjLabels.size() == 1)
        {
            int lbl = *adjLabels.begin();
            auto it = label2idx.find(lbl);
            if (it == label2idx.end()) continue;

            ZoneMask &z = allZones[it->second];

            z.mask.setTo(255, regionMask);
            occ.setTo(255, regionMask);

            ++mergedCount;
        }
    }
    return mergedCount;
}

//--------------------------------------------------------------------------
//  eraseWallsFromZones()
//--------------------------------------------------------------------------
static void eraseWallsFromZones(const cv::Mat1b &background,
                               std::vector<ZoneMask> &allZones)
{
    CV_Assert(!background.empty() && background.type() == CV_8UC1);

    for (auto &z : allZones)
    {
        CV_Assert(z.mask.size() == background.size() &&
                  z.mask.type()  == CV_8UC1);

        cv::bitwise_and(z.mask, background, z.mask);
        cv::threshold(z.mask, z.mask, 0, 255, cv::THRESH_BINARY);
    }
}

//--------------------------------------------------------------------------
//  keepCentroidComponent()
//--------------------------------------------------------------------------
static void keepCentroidComponent(const std::unordered_map<int, cv::Point> &centroids,
                                  std::vector<ZoneMask> &allZones,
                                  cv::Mat1b *occ = nullptr)
{
    if (occ)
    {
        CV_Assert(!occ->empty() && occ->type() == CV_8UC1);
    }

    for (auto &z : allZones)
    {
        auto it = centroids.find(z.label);
        if (it == centroids.end()) {
            std::cerr << "[keepCentroidComponent] no centroid for label "
                      << z.label << '\n';
            continue;
        }

        const cv::Point &c = it->second;
        cv::Mat1b &m       = z.mask;

        CV_Assert(!m.empty() && m.type() == CV_8UC1);

        if (c.x < 0 || c.x >= m.cols || c.y < 0 || c.y >= m.rows) {
            std::cerr << "[keepCentroidComponent] centroid outside image for label "
                      << z.label << '\n';
            continue;
        }

        cv::threshold(m, m, 0, 255, cv::THRESH_BINARY);

        if (m.at<uchar>(c) == 0) {
            std::cerr << "[keepCentroidComponent] centroid not inside mask for label "
                      << z.label << '\n';
            continue;
        }

        cv::Mat1i labels;
        cv::connectedComponents(m, labels, 8, CV_32S);

        int compLbl = labels.at<int>(c);

        cv::Mat1b newMask;
        cv::compare(labels, compLbl, newMask, cv::CMP_EQ);

        if (occ)
        {
            cv::Mat1b removed;
            cv::bitwise_and(m, ~newMask, removed);
            occ->setTo(0, removed);
        }

        m = newMask;
    }
}

//--------------------------------------------------------------------------
//  attachPixelsToNearestZone()
//--------------------------------------------------------------------------
static std::size_t attachPixelsToNearestZone(cv::Mat1b &occupancy,
                                             std::vector<ZoneMask> &allZones)
{
    CV_Assert(!occupancy.empty() && occupancy.type() == CV_8UC1);

    const int rows = occupancy.rows, cols = occupancy.cols;
    const int Z = static_cast<int>(allZones.size());

    std::vector<cv::Mat1f> distMaps(Z);

    for (int zi = 0; zi < Z; ++zi)
    {
        auto &z = allZones[zi];
        CV_Assert(z.mask.size() == occupancy.size() && z.mask.type() == CV_8UC1);

        cv::Mat1b invMask;
        cv::bitwise_not(z.mask, invMask);

        cv::distanceTransform(invMask,
                              distMaps[zi],
                              cv::DIST_L2,
                              3);
    }

    std::size_t attached = 0;

    for (int y = 0; y < rows; ++y)
    {
        uchar *occRow = occupancy.ptr<uchar>(y);

        for (int x = 0; x < cols; ++x)
        {
            if (occRow[x] != 0)
                continue;

            float bestDist = std::numeric_limits<float>::max();
            int   bestIdx  = -1;

            for (int zi = 0; zi < Z; ++zi)
            {
                float d = distMaps[zi].at<float>(y, x);
                if (d < bestDist)
                {
                    bestDist = d;
                    bestIdx  = zi;
                }
            }

            if (bestIdx >= 0)
            {
                allZones[bestIdx].mask(y, x) = 255;
                occRow[x]                    = 255;
                ++attached;
            }
        }
    }

    return attached;
}

//--------------------------------------------------------------------------
//  segmentByGaussianThreshold()
//--------------------------------------------------------------------------
std::vector<ZoneMask>
segmentByGaussianThreshold(const cv::Mat1b &srcBinary,
                           LabelsInfo &labelsOut,
                           int maxIter,
                           double sigmaStep,
                           double threshold)
{
    CV_Assert(!srcBinary.empty() && srcBinary.type() == CV_8UC1);

    labelsOut = LabelMapping::computeLabels(srcBinary, /*invert=*/false);

    std::unordered_map<int, cv::Point> todo = labelsOut.centroids;
    std::vector<ZoneMask> allZones;

    cv::Mat blurred, bin;

    cv::Mat1b eroded = srcBinary.clone();

    cv::Mat1b src255;
    srcBinary.convertTo(src255, CV_8U, 255);

    for (int iter = 0; iter < maxIter && !todo.empty(); ++iter)
    {
        double sigma = (iter + 1) * sigmaStep;

        cv::Mat1b kernel = cv::getStructuringElement(
                               cv::MORPH_RECT, cv::Size(2*1+1, 2*1+1));
        cv::erode(eroded, eroded, kernel, cv::Point(-1,-1), 1);

        cv::GaussianBlur(eroded, blurred, cv::Size(0,0), sigma, sigma,
                         cv::BORDER_REPLICATE);

        cv::threshold(blurred, bin, threshold, 1, cv::THRESH_BINARY);

        cv::Mat1b seg8u;
        bin.convertTo(seg8u, CV_8U, 255);

        if (cv::countNonZero(bin) == 0) {
            std::cerr << "[warn] free space vanished at iter " << iter << '\n';
            break;
        }

        auto zones = LabelMapping::extractIsolatedZones(bin, todo, /*invertFree=*/true);

        for (auto &z : zones) {
            cv::dilate(z.mask, z.mask, kernel, cv::Point(-1,-1), iter + 1);
            allZones.push_back( std::move(z) );
            todo.erase(z.label);
        }
    }

    if (!todo.empty())
    {
        cv::Mat1b occ = LabelMapping::buildOccupancyMask(src255, allZones);

        for (auto it = todo.begin(); it != todo.end(); )
        {
            if (occ(it->second) == 0)
            {
                std::cerr << "Not segmented (no zone under label): "
                          << it->first << it->second << '\n';

                ZoneMask z;
                z.label = it->first;
                z.mask  = buildFloodMask(occ, it->second);

                checkMatCompatibility(occ, z.mask);
                occ.setTo(255, z.mask);

                allZones.push_back( std::move(z) );

                it = todo.erase(it);
            }
            else
                ++it;
        }

        eraseWallsFromZones(srcBinary, allZones);

        std::size_t n = attachPixelsToNearestZone(occ, allZones);

        keepCentroidComponent(labelsOut.centroids, allZones, &occ);
        int added = mergeLonelyFreeAreas(occ, srcBinary, allZones);
        std::cout << "Добавлено участков: " << added << '\n';

        std::cerr << "[warn] maxIter reached, "
                  << todo.size() << " zones still not isolated: ";

        for (auto zone : todo)
            std::cerr << zone.first << zone.second << " ";
        std::cerr << std::endl;
    }

    return allZones;
}

