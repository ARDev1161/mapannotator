#include "map_processing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <opencv2/ximgproc.hpp>
#include <optional>
#include <string>
#include <unordered_set>

#include "mapgraph/zone_graph_dot.hpp"
#include "mapgraph/zone_graph_draw.hpp"
#include "segmentation/endpoints.h"
#include "segmentation/zoneclassifier.h"
#include "utils.hpp"

using namespace mapping;

cv::Mat1b extendWallsUntilHit(const cv::Mat1b &wallFreeMap) {
  CV_Assert(!wallFreeMap.empty() && wallFreeMap.type() == CV_8UC1);

  cv::Mat1b wallMask;
  cv::compare(wallFreeMap, 0, wallMask, cv::CMP_EQ); // 255 там, где стена

  // Скелет стены
  cv::Mat1b skeleton = getSkeletonMat(wallMask);

  // Будем дорисовывать на копии скелета: 255 — стена/скелет, 0 — фон
  cv::Mat1b result = skeleton.clone();

  // Концевые точки скелета
  std::vector<cv::Point> endpoints = findSkeletonEndpoints(skeleton);

  // Ищем прямые по скелету
  std::vector<cv::Vec4i> lines;
  cv::HoughLinesP(skeleton, lines, 1.0, CV_PI / 180.0,
                  8,   // accumulator threshold
                  4.0, // min line length
                  2.0  // max gap
  );

  // distance(endpoint, segment)
  auto pointToSegmentDist = [](const cv::Point &p, const cv::Point &a,
                               const cv::Point &b) {
    cv::Point2f ap = cv::Point2f(p - a);
    cv::Point2f ab = cv::Point2f(b - a);
    float ab2 = ab.dot(ab);
    if (ab2 < 1e-6f)
      return std::hypot(ap.x, ap.y);
    float t = std::clamp(ap.dot(ab) / ab2, 0.0f, 1.0f);
    cv::Point2f proj = cv::Point2f(a) + t * ab;
    return std::hypot(proj.x - p.x, proj.y - p.y);
  };

  // line_id -> endpoints, принадлежащие этому отрезку
  struct EndpointInfo {
    cv::Point point;
    float maxDistanceAlong = std::numeric_limits<float>::infinity();
  };
  std::unordered_map<int, std::vector<EndpointInfo>> lineEndpoints;
  const float kEndpointDist =
      1.5f; // насколько близко endpoint должен лежать к линии
  for (size_t i = 0; i < lines.size(); ++i) {
    cv::Point a(lines[i][0], lines[i][1]);
    cv::Point b(lines[i][2], lines[i][3]);
    for (const auto &ep : endpoints) {
      if (pointToSegmentDist(ep, a, b) <= kEndpointDist)
        lineEndpoints[(int)i].push_back({ep});
    }
  }

  // Направление для endpoint: продолжение линии за его пределы
  auto endpointDir = [&](const cv::Point &ep,
                         const cv::Vec4i &line) -> std::optional<cv::Point2f> {
    cv::Point p0(line[0], line[1]);
    cv::Point p1(line[2], line[3]);
    cv::Point2f dir = cv::Point2f(p1 - p0);
    float norm = std::hypot(dir.x, dir.y);
    if (norm < 1e-3f)
      return std::nullopt;
    dir *= 1.0f / norm;

    // endpoint ближе к p0 или p1 — идём «наружу» от этого конца
    float d0 = std::hypot((float)ep.x - p0.x, (float)ep.y - p0.y);
    float d1 = std::hypot((float)ep.x - p1.x, (float)ep.y - p1.y);
    if (d0 < d1)
      dir = cv::Point2f(p0 - p1) * (1.0f / norm); // наружу из p0
    else
      dir = cv::Point2f(p1 - p0) * (1.0f / norm); // наружу из p1
    return dir;
  };

  auto cross2d = [](const cv::Point2f &a, const cv::Point2f &b) {
    return a.x * b.y - a.y * b.x;
  };

  auto endpointIntersectionDistance =
      [&](int lineIdx, const EndpointInfo &info) -> float {
    auto dirOpt = endpointDir(info.point, lines[lineIdx]);
    if (!dirOpt)
      return std::numeric_limits<float>::infinity();

    cv::Point2f epPoint(info.point);
    cv::Point2f epDir = *dirOpt; // уже нормализовано

    cv::Point2f p0(lines[lineIdx][0], lines[lineIdx][1]);
    cv::Point2f p1(lines[lineIdx][2], lines[lineIdx][3]);
    cv::Point2f r = p1 - p0;
    if (std::hypot(r.x, r.y) < 1e-3f)
      return std::numeric_limits<float>::infinity();

    float best = std::numeric_limits<float>::infinity();

    for (size_t j = 0; j < lines.size(); ++j) {
      if ((int)j == lineIdx)
        continue;

      cv::Point2f q0(lines[j][0], lines[j][1]);
      cv::Point2f q1(lines[j][2], lines[j][3]);
      cv::Point2f s = q1 - q0;
      if (std::hypot(s.x, s.y) < 1e-3f)
        continue;

      float denom = cross2d(r, s);
      if (std::abs(denom) < 1e-4f)
        continue; // параллельные или слишком близко к этому

      cv::Point2f qmp = q0 - p0;
      float t = cross2d(qmp, s) / denom;
      cv::Point2f inter = p0 + t * r;

      float alongDir = (inter - epPoint).dot(epDir);
      if (alongDir <= 0.0f)
        continue; // пересечение позади роста

      if (alongDir < best)
        best = alongDir;
    }

    return best;
  };

  for (auto &[lineIdx, eps] : lineEndpoints) {
    for (auto &info : eps)
      info.maxDistanceAlong = endpointIntersectionDistance(lineIdx, info);
  }

  // Активные концы: растём синхронно на 1 пиксель за итерацию
  struct ActiveEndpoint {
    cv::Point2f pos;
    cv::Point2f dir;
    cv::Point2f start;
    float maxDistance;
    int lineIdx;
  };
  std::vector<ActiveEndpoint> active;
  for (size_t i = 0; i < lines.size(); ++i) {
    auto it = lineEndpoints.find((int)i);
    if (it == lineEndpoints.end() || it->second.empty())
      continue;
    for (const auto &epInfo : it->second) {
      auto dirOpt = endpointDir(epInfo.point, lines[i]);
      if (!dirOpt)
        continue;
      active.push_back({cv::Point2f(epInfo.point),
                        *dirOpt,
                        cv::Point2f(epInfo.point),
                        epInfo.maxDistanceAlong,
                        static_cast<int>(i)});
    }
  }

  const int maxSteps = result.rows + result.cols;
  for (int step = 0; step < maxSteps && !active.empty(); ++step) {
    std::vector<size_t> toRemove;
    toRemove.reserve(active.size());

    for (size_t idx = 0; idx < active.size(); ++idx) {
      auto &ae = active[idx];
      cv::Point2f next = ae.pos + ae.dir;
      float nextAlong = (next - ae.start).dot(ae.dir);
      if (nextAlong > ae.maxDistance + 1e-3f) {
        toRemove.push_back(idx);
        continue;
      }
      int x = cvRound(next.x);
      int y = cvRound(next.y);
      if (x < 0 || y < 0 || x >= result.cols || y >= result.rows) {
        toRemove.push_back(idx);
        continue;
      }

      if (result(y, x) == 0) {
        result(y, x) = 255; // рисуем 1 пиксель за итерацию
        ae.pos = next;
      } else {
        toRemove.push_back(idx); // упёрлись в белое
      }
    }

    if (!toRemove.empty()) {
      std::sort(toRemove.rbegin(), toRemove.rend());
      for (size_t idx : toRemove) {
        active[idx] = active.back();
        active.pop_back();
      }
    }
  }

  return result;
}

//--------------------------------------------------------------------------
//  buildZoneAdjacency()
//--------------------------------------------------------------------------
static std::unordered_map<int, std::vector<int>>
buildZoneAdjacency(const cv::Mat1i &zones, bool use8Connectivity = false) {
  CV_Assert(!zones.empty() && zones.type() == CV_32S);

  const cv::Size sz = zones.size();

  const std::vector<cv::Point> off4{{1, 0}, {0, 1}};
  const std::vector<cv::Point> off8{{1, 0}, {0, 1}, {1, 1}, {1, -1}};
  const auto &offs = use8Connectivity ? off8 : off4;

  std::unordered_map<int, std::unordered_set<int>> tmpAdj;

  for (int y = 0; y < sz.height; ++y) {
    const int *row = zones.ptr<int>(y);
    for (int x = 0; x < sz.width; ++x) {
      int lbl = row[x];
      if (lbl <= 0)
        continue; // фон

      for (const auto &d : offs) {
        int nx = x + d.x, ny = y + d.y;
        if ((unsigned)nx >= (unsigned)sz.width ||
            (unsigned)ny >= (unsigned)sz.height)
          continue;

        int nLbl = zones.at<int>(ny, nx);
        if (nLbl <= 0 || nLbl == lbl)
          continue; // тот же или фон

        tmpAdj[lbl].insert(nLbl);
        tmpAdj[nLbl].insert(lbl); // граф неориентированный
      }
    }
  }

  std::unordered_map<int, std::vector<int>> adjacency;
  adjacency.reserve(tmpAdj.size());

  for (auto &[lbl, set] : tmpAdj) {
    adjacency[lbl] = {set.begin(), set.end()};
    std::sort(adjacency[lbl].begin(), adjacency[lbl].end());
  }
  return adjacency;
}

static cv::Vec3b labelColor(int label) {
  cv::RNG rng(static_cast<uint64_t>(label) * 9781u + 13579u);
  return cv::Vec3b(rng.uniform(60, 255), rng.uniform(60, 255),
                   rng.uniform(60, 255));
}

cv::Mat renderZonesOverlay(const std::vector<ZoneMask> &zones,
                           const cv::Mat &baseImage,
                           const Segmentation::CropInfo &cropInfo,
                           double alpha) {
  CV_Assert(!baseImage.empty());

  cv::Mat baseColor;
  if (baseImage.channels() == 1)
    cv::cvtColor(baseImage, baseColor, cv::COLOR_GRAY2BGR);
  else if (baseImage.channels() == 3)
    baseColor = baseImage.clone();
  else
    baseColor = baseImage.clone();

  cv::Rect roi(cropInfo.left, cropInfo.top,
               baseColor.cols - cropInfo.left - cropInfo.right,
               baseColor.rows - cropInfo.top - cropInfo.bottom);

  CV_Assert(roi.width > 0 && roi.height > 0);
  cv::Mat roiView = baseColor(roi);
  cv::Mat tinted = roiView.clone();

  for (const auto &zone : zones) {
    CV_Assert(zone.mask.size() == roi.size());
    tinted.setTo(labelColor(zone.label), zone.mask);
  }

  alpha = std::clamp(alpha, 0.0, 1.0);
  cv::addWeighted(roiView, 1.0 - alpha, tinted, alpha, 0.0, tinted);
  return tinted;
}

//--------------------------------------------------------------------------
//  buildZoneIndex()
//--------------------------------------------------------------------------
static std::unordered_map<int, const cv::Mat1b *>
buildZoneIndex(const std::vector<ZoneMask> &zones) {
  std::unordered_map<int, const cv::Mat1b *> index;
  index.reserve(zones.size());

  for (const auto &z : zones)
    index.emplace(z.label, &z.mask);

  return index;
}

//--------------------------------------------------------------------------
//  annotateGraph()
//--------------------------------------------------------------------------
static void annotateGraph(ZoneGraph &graph, const std::string &rulefile) {
  static ZoneClassifier clf{rulefile};

  for (auto &node : graph.allNodes()) {
    ZoneFeatures f = node->features();
    ZoneType t = clf.classify(f);

    node->setType(t);
    std::cerr << "Zone #" << node->id() << " → " << t.path() << '\n';
  }
}

//--------------------------------------------------------------------------
//  buildGraph()
//--------------------------------------------------------------------------
void buildGraph(ZoneGraph &graphOut, std::vector<ZoneMask> zones,
                cv::Mat1i zonesMat, const MapInfo &mapParams,
                std::unordered_map<int, cv::Point> centroids) {
  std::unordered_map<ZoneType, NodePtr> node; // для удобства соединений

  auto add = [&](ZoneId id, ZoneType t, cv::Point2d c, double area,
                 double perimeter) {
    node[t] = graphOut.addNode(id, t, area, perimeter, c,
                               {{{c.x - 0.5, c.y - 0.5},
                                 {c.x + 0.5, c.y - 0.5},
                                 {c.x + 0.5, c.y + 0.5},
                                 {c.x - 0.5, c.y + 0.5}}});
  };

  auto W = [](double m) { return m; }; // удобный alias

  std::unordered_map<int, std::vector<int>> zoneGraph =
      buildZoneAdjacency(zonesMat, /*use8Connectivity=*/true);

  // Assume map resolution is 5 cm per pixel ⇒ area per pixel = 0.05 × 0.05 m².
  constexpr double kAreaPerPixel = 0.05 * 0.05;

  auto zoneIndex = buildZoneIndex(zones);

  // Создаем ноды графа
  for (auto centroid : centroids) {
    // размещаем вершины (координаты в «метрах»)
    add(centroid.first, ZoneType{}, pixelToWorld(centroid.second, mapParams),
        computeWhiteArea(zoneIndex[centroid.first], kAreaPerPixel), 10);

    // строим рёбра
    std::cerr << "Zone " << std::to_string(centroid.first) << centroid.second
              << " neighbours:";
    for (auto neighbour : zoneGraph[centroid.first]) {
      std::cerr << ' ' << neighbour;
      graphOut.connectZones(centroid.first, neighbour, W(1.4));
    }
    std::cerr << '\n';
  }

  auto locateRules = []() -> std::string {
    namespace fs = std::filesystem;
    if (const char *env = std::getenv("MAPANNOTATOR_RULES_PATH")) {
      fs::path env_path(env);
      if (fs::exists(env_path))
        return env_path.string();
    }
    const std::vector<fs::path> candidates = {
        fs::path("config") / "rules.yaml", fs::path("../config") / "rules.yaml",
        fs::path("share") / "mapannotator" / "config" / "rules.yaml"};
    for (const auto &p : candidates) {
      if (fs::exists(p))
        return p.string();
    }
    if (const char *ament_prefix = std::getenv("AMENT_PREFIX_PATH")) {
      std::string prefixes(ament_prefix);
      size_t start = 0;
      while (start <= prefixes.size()) {
        size_t end = prefixes.find(':', start);
        std::string prefix = prefixes.substr(start, end - start);
        if (!prefix.empty()) {
          fs::path candidate = fs::path(prefix) / "share" / "mapannotator" / "config" / "rules.yaml";
          if (fs::exists(candidate))
            return candidate.string();
        }
        if (end == std::string::npos)
          break;
        start = end + 1;
      }
    }
    return (fs::path("config") / "rules.yaml").string();
  };

  annotateGraph(graphOut, locateRules());

  /* ---------- DOT‑файл ----------------------------------------------- */
  std::ofstream dot("graph.dot");
  writeDot(graphOut, dot);

  /* ---------- Быстрый предварительный просмотр (OpenCV) -------------- */
  cv::Mat img;
  drawZoneGraph(graphOut, img, 50.0 /*px/м*/, 7 /*радиус*/,
                true /*подписи ширины*/, false /*invertY*/,
                4000 /*max canvas px*/);
  showMatDebug("Floor‑Plan Graph", img);
  cv::imwrite("graph_preview.png", img);
}

//--------------------------------------------------------------------------
//  buildFloodMask()
//--------------------------------------------------------------------------
static cv::Mat1b buildFloodMask(const cv::Mat1b &freeMap, const cv::Point &seed,
                                int connectivity = 8) {
  CV_Assert(freeMap.type() == CV_8UC1);
  CV_Assert(seed.x >= 0 && seed.x < freeMap.cols && seed.y >= 0 &&
            seed.y < freeMap.rows);

  if (freeMap(seed) != 0)
    return cv::Mat1b::zeros(freeMap.size());

  cv::Mat1b mask(freeMap.rows + 2, freeMap.cols + 2, uchar(0));

  const int flags = connectivity | cv::FLOODFILL_MASK_ONLY | (255 << 8);
  cv::floodFill(freeMap, mask, seed, 0, nullptr, cv::Scalar(), cv::Scalar(),
                flags);

  return mask(cv::Rect(1, 1, freeMap.cols, freeMap.rows)).clone();
}

//--------------------------------------------------------------------------
//  mergeLonelyFreeAreas()
//--------------------------------------------------------------------------
static int mergeLonelyFreeAreas(cv::Mat1b &occ, const cv::Mat1b &background,
                                std::vector<ZoneMask> &allZones) {
  CV_Assert(occ.size() == background.size() && occ.type() == CV_8UC1 &&
            background.type() == CV_8UC1);

  cv::Mat1i labelMap = cv::Mat1i::zeros(occ.size());
  std::unordered_map<int, std::size_t> label2idx;

  for (std::size_t i = 0; i < allZones.size(); ++i) {
    const auto &z = allZones[i];
    CV_Assert(z.mask.size() == occ.size() && z.mask.type() == CV_8UC1);
    labelMap.setTo(z.label, z.mask);
    label2idx[z.label] = i;
  }

  cv::Mat1b freeMask;
  cv::bitwise_and(occ == 0, background, freeMask);

  cv::Mat1i compLabels;
  int nComp = cv::connectedComponents(freeMask, compLabels, 8, CV_32S);

  const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
  int mergedCount = 0;

  for (int id = 1; id < nComp; ++id) {
    cv::Mat1b regionMask = (compLabels == id);

    cv::Mat1b dilated;
    cv::dilate(regionMask, dilated, kernel, {-1, -1}, 1);

    std::vector<cv::Point> pts;
    cv::findNonZero(dilated, pts);

    std::unordered_set<int> adjLabels;
    for (const auto &p : pts) {
      int lbl = labelMap.at<int>(p);
      if (lbl > 0)
        adjLabels.insert(lbl);
    }

    if (adjLabels.size() == 1) {
      int lbl = *adjLabels.begin();
      auto it = label2idx.find(lbl);
      if (it == label2idx.end())
        continue;

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
                                std::vector<ZoneMask> &allZones) {
  CV_Assert(!background.empty() && background.type() == CV_8UC1);

  for (auto &z : allZones) {
    CV_Assert(z.mask.size() == background.size() && z.mask.type() == CV_8UC1);

    cv::bitwise_and(z.mask, background, z.mask);
    cv::threshold(z.mask, z.mask, 0, 255, cv::THRESH_BINARY);
  }
}

//--------------------------------------------------------------------------
//  keepCentroidComponent()
//--------------------------------------------------------------------------
static void
keepCentroidComponent(const std::unordered_map<int, cv::Point> &centroids,
                      std::vector<ZoneMask> &allZones,
                      cv::Mat1b *occ = nullptr) {
  if (occ) {
    CV_Assert(!occ->empty() && occ->type() == CV_8UC1);
  }

  for (auto &z : allZones) {
    auto it = centroids.find(z.label);
    if (it == centroids.end()) {
      std::cerr << "[keepCentroidComponent] no centroid for label " << z.label
                << '\n';
      continue;
    }

    const cv::Point &c = it->second;
    cv::Mat1b &m = z.mask;

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

    if (occ) {
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
                                             std::vector<ZoneMask> &allZones) {
  CV_Assert(!occupancy.empty() && occupancy.type() == CV_8UC1);

  const int rows = occupancy.rows, cols = occupancy.cols;
  const int Z = static_cast<int>(allZones.size());

  // ownerMap keeps the index of the zone that currently owns the pixel; -1 =
  // free/unassigned.
  cv::Mat1i ownerMap(occupancy.rows, occupancy.cols, int(-1));
  std::deque<cv::Point> queue;
  queue.clear();

  std::vector<cv::Point> tmpPts;
  tmpPts.reserve(rows * cols / 8);

  // Seed the BFS with all zone pixels.
  for (int zi = 0; zi < Z; ++zi) {
    auto &z = allZones[zi];
    CV_Assert(z.mask.size() == occupancy.size() && z.mask.type() == CV_8UC1);

    tmpPts.clear();
    cv::findNonZero(z.mask, tmpPts);
    for (const auto &p : tmpPts) {
      ownerMap(p) = zi;
      queue.push_back(p);
    }
  }

  const std::array<cv::Point, 8> neighbors = {
      cv::Point{1, 0}, cv::Point{-1, 0}, cv::Point{0, 1},  cv::Point{0, -1},
      cv::Point{1, 1}, cv::Point{1, -1}, cv::Point{-1, 1}, cv::Point{-1, -1}};

  std::size_t attached = 0;

  while (!queue.empty()) {
    const cv::Point p = queue.front();
    queue.pop_front();
    int ownerIdx = ownerMap(p);
    if (ownerIdx < 0 || ownerIdx >= Z)
      continue;

    for (const auto &d : neighbors) {
      int nx = p.x + d.x;
      int ny = p.y + d.y;
      if ((unsigned)nx >= (unsigned)cols || (unsigned)ny >= (unsigned)rows)
        continue;

      if (occupancy(ny, nx) != 0) // already assigned or wall/zone
        continue;

      if (ownerMap(ny, nx) != -1)
        continue;

      ownerMap(ny, nx) = ownerIdx;
      occupancy(ny, nx) = 255;
      allZones[ownerIdx].mask(ny, nx) = 255;
      queue.push_back({nx, ny});
      ++attached;
    }
  }

  return attached;
}

//--------------------------------------------------------------------------
//  segmentByGaussianThreshold()
//--------------------------------------------------------------------------
static std::unordered_map<int, cv::Point>
computeZoneCentroids(const std::vector<ZoneMask> &zones) {
  std::unordered_map<int, cv::Point> centroids;
  for (const auto &zone : zones) {
    cv::Moments m = cv::moments(zone.mask, true);
    if (m.m00 <= 0.0)
      continue;
    int cx = static_cast<int>(std::round(m.m10 / m.m00));
    int cy = static_cast<int>(std::round(m.m01 / m.m00));
    cx = std::clamp(cx, 0, zone.mask.cols - 1);
    cy = std::clamp(cy, 0, zone.mask.rows - 1);
    centroids[zone.label] = cv::Point(cx, cy);
  }
  return centroids;
}

static LabelsInfo buildLabelsFromZones(const std::vector<ZoneMask> &zones) {
  LabelsInfo info;
  if (zones.empty())
    return info;

  info.centroidLabels = cv::Mat1i::zeros(zones.front().mask.size());
  info.numLabels = static_cast<int>(zones.size()) + 1;

  for (const auto &zone : zones) {
    cv::Moments m = cv::moments(zone.mask, true);
    if (m.m00 <= 0.0)
      continue;
    int cx = static_cast<int>(std::round(m.m10 / m.m00));
    int cy = static_cast<int>(std::round(m.m01 / m.m00));
    cx = std::clamp(cx, 0, zone.mask.cols - 1);
    cy = std::clamp(cy, 0, zone.mask.rows - 1);
    info.centroids[zone.label] = cv::Point(cx, cy);
    info.centroidLabels(cy, cx) = zone.label;
  }
  return info;
}

std::vector<ZoneMask>
segmentByGaussianThreshold(const cv::Mat1b &srcBinary, LabelsInfo &labels,
                           const SegmentationParams &params) {
  CV_Assert(!srcBinary.empty() && srcBinary.type() == CV_8UC1);

  // Normalise to 0/255 for algorithms that assume 8U maps in that range
  // (down-sample seeding, occupancy masks). The legacy branch still works on
  // the original 0/1 map so we keep both representations.
  double minVal = 0.0, maxVal = 0.0;
  cv::minMaxLoc(srcBinary, &minVal, &maxVal);
  const double to255 = (maxVal <= 1.0) ? 255.0 : 1.0;
  cv::Mat1b src255;
  srcBinary.convertTo(src255, CV_8U, to255);

  // todo holds centroids that still need a separate zone; allZones accumulates
  // results.
  std::unordered_map<int, cv::Point> todo = labels.centroids;
  std::vector<ZoneMask> allZones;

  cv::Mat blurred, bin;

  cv::Mat1b eroded = srcBinary.clone();

  const int stallLimit = 5;
  int prevTodoSize = static_cast<int>(todo.size());
  int stallIters = 0;

  // Small structuring element used both for erosion of the free space
  // and for dilating masks back after a zone is extracted.
  cv::Mat1b kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * 1 + 1, 2 * 1 + 1));

  for (int iter = 0; iter < params.maxIter && !todo.empty(); ++iter) {
    double sigma = (iter + 1) * params.sigmaStep;

    // Erode obstacles a bit to avoid leaking through narrow gaps,
    // blur the binary map, then threshold to obtain a softened free-space mask.
    cv::erode(eroded, eroded, kernel, cv::Point(-1, -1), 1);

    cv::GaussianBlur(eroded, blurred, cv::Size(0, 0), sigma, sigma,
                     cv::BORDER_REPLICATE);

    cv::threshold(blurred, bin, params.threshold, 1, cv::THRESH_BINARY);

    cv::Mat1b seg8u;
    bin.convertTo(seg8u, CV_8U, 255);

    if (cv::countNonZero(bin) == 0) {
      std::cerr << "[warn] free space vanished at iter " << iter << '\n';
      break;
    }

    // Extract zones containing the outstanding centroids;
    // dilate them back to compensate for the prior erosion/blur.
    auto zones =
        LabelMapping::extractIsolatedZones(bin, todo, /*invertFree=*/true);

    for (auto &z : zones) {
      cv::dilate(z.mask, z.mask, kernel, cv::Point(-1, -1), iter + 1);
      allZones.push_back(std::move(z));
      todo.erase(z.label);
      stallIters = 0;
    }

    const int curTodo = static_cast<int>(todo.size());
    if (curTodo == prevTodoSize)
      ++stallIters;
    else
      stallIters = 0;
    prevTodoSize = curTodo;

    if (stallIters >= stallLimit) {
      std::cerr << "[warn] segmentation stalled for " << stallIters
                << " iterations, breaking early with " << curTodo
                << " zones pending\n";
      break;
    }
  }

  // If some centroids never got isolated, flood-fill their component and
  // then run the same cleanup/attachment steps as the fast path.
  if (!todo.empty()) {
      cv::Mat out;
      cv::Mat1b occ = LabelMapping::buildOccupancyMask(src255, allZones);
      cv::Mat1b extended = extendWallsUntilHit(srcBinary);

      // Добавляем продлённые стены в маску занятых
      cv::bitwise_or(occ, extended, occ);

      occ.convertTo(out, CV_8U, 255);
      showMat("Extended", out);

      for (auto it = todo.begin(); it != todo.end();) {
          if (occ(it->second) == 0) {
              std::cerr << "Not segmented (no zone under label): " << it->first << it->second
                        << '\n';

              ZoneMask z;
              z.label = it->first;
              z.mask = buildFloodMask(occ, it->second);

              checkMatCompatibility(occ, z.mask);
              occ.setTo(255, z.mask);

              allZones.push_back(std::move(z));

              it = todo.erase(it);
          } else
              ++it;
      }

    eraseWallsFromZones(srcBinary, allZones);

    // std::size_t n = attachPixelsToNearestZone(occ, allZones);
    //     (void)n;

    // keepCentroidComponent(labels.centroids, allZones, &occ);
    // int added = mergeLonelyFreeAreas(occ, srcBinary, allZones);
    // std::cout << "Добавлено участков: " << added << '\n';

    std::cerr << "[warn] maxIter reached, " << todo.size() << " zones still not isolated: ";

    for (auto zone : todo)
        std::cerr << zone.first << zone.second << " ";
    std::cerr << std::endl;
  }

  labels = buildLabelsFromZones(allZones);
  return allZones;
}
