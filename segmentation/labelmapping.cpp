#include "labelmapping.hpp"
#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>

int LabelMapping::addCornerCirclesHarris(cv::Mat1b &wallMask,
                                         int radius,
                                         uchar circleVal,
                                         int maxCorners,
                                         double harrisK,
                                         double quality,
                                         double minDist)
{
    CV_Assert(!wallMask.empty() && wallMask.type() == CV_8UC1);
    CV_Assert(radius > 1);

    // Shi–Tomasi (goodFeaturesToTrack) ищет углы.
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(wallMask,
                            corners,
                            maxCorners,
                            quality,
                            std::max<double>(minDist, radius),
                            cv::noArray(),
                            3,    // blockSize
                            true, // use Harris
                            harrisK);

    /* ---------- 3. Рисуем окружности прямо в wallMask ---------- */
    int drawn = 0;
    for (const auto &p : corners) {
        cv::circle(wallMask, p, radius, circleVal, -1, cv::LINE_AA); // -1 → заполненный круг
        ++drawn;
    }
    return drawn;
}

cv::Mat LabelMapping::getLocalMax(const cv::Mat &dist)
{
    // Ищем локальные максимумы. Для этого применяем дилатацию.
    cv::Mat dilated;
    cv::dilate(dist, dilated, cv::Mat());

    // Локальный максимум там, где значение равно дилатированному.
    cv::Mat localMax;
    cv::compare(dist, dilated, localMax, cv::CMP_EQ);

    // Отфильтровываем слабые максимумы: оставляем только те, что превышают порог.
    double minVal, maxVal;
    cv::minMaxLoc(dist, &minVal, &maxVal);
    cv::Mat strongMax;
    // Например, оставляем локальные максимумы, превышающие 10% от максимального значения.
    cv::threshold(dist, strongMax, 0.1 * maxVal, 1.0, cv::THRESH_BINARY);

    // Объединяем маску локальных максимумов с маской сильных пиков.
    cv::Mat peaks;
    cv::Mat strongMax8U;
    strongMax.convertTo(strongMax8U, CV_8U, 255); // масштабируем значения к диапазону [0,255]
    cv::bitwise_and(localMax, strongMax8U, peaks);

    // Приводим peaks к 8-битному типу (маска: 0 или 255)
    peaks.convertTo(peaks, CV_8U, 255);

    return peaks;
}

inline LabelsInfo LabelMapping::getCentroids(const cv::Mat1b &binaryImage, bool invert)
{
    // Проверяем, что входное изображение имеет тип CV_8UC1
    CV_Assert(binaryImage.type() == CV_8UC1);

    // ── 1. Гарантируем «фон = 0» ─────────────────────────────────────────────
    cv::Mat1b mask;
    if (invert)
        cv::bitwise_not(binaryImage, mask);
    else
        mask = binaryImage;

    // ── 2.   Connected Components + центроиды ───────────────────────────────
    cv::Mat1i labels, stats;
    cv::Mat1d centers; // rows = numLabels, cols = 2 (x,y)
    int nLabels = cv::connectedComponentsWithStats(mask, labels, stats, centers, 8, CV_32S);

    // ── 3.   Sparse‑карта центроидов (CV_32S) ───────────────────────────────
    cv::Mat1i centLbl = cv::Mat::zeros(mask.size(), CV_32S);
    std::unordered_map<int, cv::Point> cmap;

    for (int lbl = 1; lbl < nLabels; ++lbl) { // label 0 = фон
        int cx = static_cast<int>(std::round(centers(lbl, 0)));
        int cy = static_cast<int>(std::round(centers(lbl, 1)));

        // защита от округления за пределы
        cx = std::clamp(cx, 0, centLbl.cols - 1);
        cy = std::clamp(cy, 0, centLbl.rows - 1);

        centLbl(cy, cx) = lbl; // код метки в центроиде
        cmap.emplace(lbl, cv::Point(cx, cy));
    }

    return {std::move(centLbl), std::move(cmap), nLabels};
}

LabelsInfo LabelMapping::getSeeds(const cv::Mat &dist, int dilateIterations, int dilateKernelSize)
{
    cv::Mat labels, dilated;

    labels = getLocalMax(dist);
    showMatDebug("Local Max", labels);

    dilated = dilateBinary(
        labels,
        dilateKernelSize,
        dilateIterations); // расширяем для слияния близкорасположенных локальных максимумов
    showMatDebug("Dilated max", dilated);

    LabelsInfo labelInfo = getCentroids(dilated); // получаем центры областей локальных максимумов
    return labelInfo;
}

bool LabelMapping::obstacleWithinRadius(const cv::Mat1b &binaryMap,
                                        const cv::Point &centroid,
                                        int radiusPx)
{
    if (radiusPx <= 0)
        return false;

    const int radiusSq = radiusPx * radiusPx;
    const int x0 = std::max(0, centroid.x - radiusPx);
    const int x1 = std::min(binaryMap.cols - 1, centroid.x + radiusPx);
    const int y0 = std::max(0, centroid.y - radiusPx);
    const int y1 = std::min(binaryMap.rows - 1, centroid.y + radiusPx);

    for (int y = y0; y <= y1; ++y) {
        const uchar *row = binaryMap.ptr<uchar>(y);
        for (int x = x0; x <= x1; ++x) {
            if (row[x] != 0)
                continue;
            const int dx = x - centroid.x;
            const int dy = y - centroid.y;
            if (dx * dx + dy * dy <= radiusSq)
                return true;
        }
    }
    return false;
}

void LabelMapping::filterSeedCentroids(LabelsInfo &labels,
                                       const cv::Mat1b &binaryMap,
                                       double minDistancePx)
{
    if (minDistancePx <= 0.0 || labels.centroids.empty())
        return;

    const int radiusPx = std::max(0, static_cast<int>(std::ceil(minDistancePx)));
    if (radiusPx <= 0)
        return;

    int removed = 0;
    for (auto it = labels.centroids.begin(); it != labels.centroids.end();) {
        if (it->first <= 0 || it->second.x < 0 || it->second.y < 0 || it->second.x >= binaryMap.cols
            || it->second.y >= binaryMap.rows) {
            ++it;
            continue;
        }

        if (obstacleWithinRadius(binaryMap, it->second, radiusPx)) {
            labels.centroidLabels(it->second) = 0;
            it = labels.centroids.erase(it);
            ++removed;
        } else {
            ++it;
        }
    }

    if (removed > 0) {
        labels.numLabels = static_cast<int>(labels.centroids.size()) + 1;
        std::cout << "[info] Removed " << removed << " centroid seed(s) closer than "
                  << minDistancePx << "px to obstacles\n";
    }
}

LabelsInfo LabelMapping::computeLabels(const cv::Mat1b &binaryDilated,
                                       int backgroundErosionKernelSize,
                                       double seedClearancePx)
{
    // Создаем маску фона (эрозия бинарной карты).
    cv::Mat kernel = cv::Mat::ones(backgroundErosionKernelSize, backgroundErosionKernelSize, CV_8U);
    cv::Mat eroded;
    cv::erode(binaryDilated, eroded, kernel);
    cv::Mat1b backgroundMask = (eroded == 1);
    cv::dilate(binaryDilated, eroded, kernel);

    // Находим углы и расширяем их чтобы повысить количество локальных максимумов если некоторые зоны не достаточно отделены
    addCornerCirclesHarris(backgroundMask, 4);

    // TODO: сделать синхронное продление стен(находим прямые в binaryDilated и
    // продляем их до первого черного пиксела и сохраняем в отдельный Mat possibleWalls
    // после чего в backgroundMask поэтапно наращиваем стены по маске possibleWalls)
    // это надо делать в сегментации

    // Perform the distance transform algorithm
    cv::Mat dist;
    cv::distanceTransform(backgroundMask, dist, cv::DIST_L2, cv::DIST_MASK_3);
    cv::Mat mask = (binaryDilated == 0);
    dist.setTo(-1, mask);

    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);

    LabelsInfo labels = getSeeds(dist, 2, 9);

    labels.centroidLabels.setTo(-1, backgroundMask == 0); ///< Remove centroids located into walls

    cv::Mat outCentroids;
    labels.centroidLabels.convertTo(outCentroids, CV_8U, 255);
    showMat("Centroids", outCentroids);

    showMat("backgroundMask", backgroundMask);

    if (seedClearancePx != 0.0)
        LabelMapping::filterSeedCentroids(labels, binaryDilated, seedClearancePx);

    return labels;
}

std::vector<ZoneMask> LabelMapping::extractIsolatedZones(
    const cv::Mat1b &freeMap, const std::unordered_map<int, cv::Point> &centroids, bool invertFree)
{
    CV_Assert(!freeMap.empty() && freeMap.type() == CV_8UC1);

    /* ─ 1. Маска «свободно = 255» ──────────────────────────────────────────── */
    cv::Mat1b walkMask;
    invertFree ? cv::compare(freeMap, 0, walkMask, cv::CMP_NE)  // ≠0 → 255
               : cv::compare(freeMap, 0, walkMask, cv::CMP_EQ); //  0 → 255

    /* ─ 2. Размечаем компоненты свободного пространства ───────────────────── */
    cv::Mat1i comp; // CV_32S
    int nComp = cv::connectedComponents(walkMask, comp, 8, CV_32S);

    /* ─ 3. Считаем, сколько центроидов в каждой компоненте ────────────────── */
    std::unordered_map<int, std::vector<int>> bindings; // compID → labels
    int backgroundLabel = 0;                            // 0 = фон/стены
    for (auto &[label, coord] : centroids) {
        if (label == backgroundLabel) // 0 — стены/фон, пропускаем
            continue;
        int compId = comp(coord);
        if (compId != backgroundLabel)
            bindings[compId].push_back(label);
    }

    /* ─ 4. Формируем зоны для оставшихся меток ───────────── */
    std::vector<ZoneMask> zones;
    zones.reserve(bindings.size());
    for (const auto &[compId, labels] : bindings) {
        if (labels.size() == 1) // проверка изолированности зоны
        {
            // Создаём бинарную маску: 255 там, где comp == compId
            cv::Mat1b mask;
            cv::compare(comp, compId, mask, cv::CMP_EQ); // результат 8-бит, 0 / 255

            zones.push_back({labels.front(), std::move(mask)});
        }
    }

    return zones;
}

cv::Mat1b LabelMapping::buildOccupancyMask(const cv::Mat1b &background,
                                           const std::vector<ZoneMask> &allZones)
{
    CV_Assert(!background.empty() && background.type() == CV_8UC1);

    /* 1. Начинаем с копии background; сюда «нагружаем» зоны */
    cv::Mat1b occ = ~background.clone(); // 0 = стена, 255 = свободно

    /* 2. Каждую зону превращаем в занятое пространство */
    for (const auto &z : allZones) {
        CV_Assert(z.mask.size() == background.size() && z.mask.type() == CV_8UC1);

        // Белые пиксели (255) из z.mask копируем в occ
        occ.setTo(255, z.mask);
    }

    return occ;
}
