#include "labelmapping.hpp"
#include <set>
#include <algorithm>
#include <numeric>
#include <stdexcept>

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
    cv::Mat1d centers;                       // rows = numLabels, cols = 2 (x,y)
    int nLabels = cv::connectedComponentsWithStats(
                      mask, labels, stats, centers, 8, CV_32S);

    // ── 3.   Sparse‑карта центроидов (CV_32S) ───────────────────────────────
    cv::Mat1i centLbl = cv::Mat::zeros(mask.size(), CV_32S);
    std::unordered_map<int, cv::Point> cmap;

    for (int lbl = 1; lbl < nLabels; ++lbl) {          // label 0 = фон
        int cx = static_cast<int>(std::round(centers(lbl, 0)));
        int cy = static_cast<int>(std::round(centers(lbl, 1)));

        // защита от округления за пределы
        cx = std::clamp(cx, 0, centLbl.cols - 1);
        cy = std::clamp(cy, 0, centLbl.rows - 1);

        centLbl(cy, cx) = lbl;                         // код метки в центроиде
        cmap.emplace(lbl, cv::Point(cx, cy));
    }

    return { std::move(centLbl), std::move(cmap), nLabels };
}

LabelsInfo LabelMapping::getSeeds(const cv::Mat &dist, int dilateIterations, int dilateKernelSize)
{
    cv::Mat labels, dilated;

    showMat("Dist", dist);

    labels = getLocalMax(dist);
    showMat("Local Max", labels);

    dilated = dilateBinary(labels, dilateKernelSize, dilateIterations); // расширяем для слияния близкорасположенных локальных максимумов
    showMat("Dilated", dilated);

    LabelsInfo labelInfo = getCentroids(dilated); // получаем центры областей локальных максимумов

    return labelInfo;
}

LabelsInfo LabelMapping::computeLabels(const cv::Mat& binaryDilated, int backgroundErosionKernelSize)
{
    // Создаем маску фона (эрозия бинарной карты).
    cv::Mat kernel = cv::Mat::ones(backgroundErosionKernelSize, backgroundErosionKernelSize, CV_8U);
    cv::Mat eroded;
    cv::erode(binaryDilated, eroded, kernel);
    cv::Mat backgroundMask = (eroded == 1);
    cv::dilate(binaryDilated, eroded, kernel);

    // Perform the distance transform algorithm
    cv::Mat dist;
    cv::distanceTransform(backgroundMask, dist, cv::DIST_L2, cv::DIST_MASK_3);
    cv::Mat mask = (binaryDilated == 0);
    dist.setTo(-1, mask);

    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);

    LabelsInfo labels = getSeeds(dist, 2, 9);
    labels.centroidLabels.setTo(-1, backgroundMask == 0);

    showMat("backgroundMask", backgroundMask);

    cv::Mat outLabels;
    labels.centroidLabels.convertTo(outLabels, CV_8U, 255);
    cv::imshow("labels", outLabels);

    return labels;
}

std::vector<ZoneMask>
LabelMapping::extractIsolatedZones(const cv::Mat1b& freeMap,
                     const std::unordered_map<int, cv::Point>& centroids,
                     bool invertFree)
{
    CV_Assert(!freeMap.empty() && freeMap.type() == CV_8UC1);

    /* ─ 1. Маска «свободно = 255» ──────────────────────────────────────────── */
    cv::Mat1b walkMask;
    invertFree ? cv::compare(freeMap, 0, walkMask, cv::CMP_NE)   // ≠0 → 255
               : cv::compare(freeMap, 0, walkMask, cv::CMP_EQ);  //  0 → 255

    /* ─ 2. Размечаем компоненты свободного пространства ───────────────────── */
    cv::Mat1i comp;                           // CV_32S
    int nComp = cv::connectedComponents(walkMask, comp, 8, CV_32S);

    /* ─ 3. Считаем, сколько центроидов в каждой компоненте ────────────────── */
    std::vector<int> count(nComp, 0);
    std::unordered_map<int,int> centroid2comp;    // label → compID

    for (auto& [lbl, pt] : centroids) {
        int cid = 0;                              // 0 = фон/за пределами
        if (pt.inside(cv::Rect(0,0,freeMap.cols,freeMap.rows)))
            cid = comp(pt);
        centroid2comp[lbl] = cid;
        if (cid > 0) ++count[cid];
    }

    /* ─ 4. Формируем маски только для изолированных компонентов ───────────── */
    std::vector<ZoneMask> zones;
    cv::Mat1b tmpMask(freeMap.size(), 0);         // будет переиспользоваться

    for (auto& [lbl, cid] : centroid2comp)
        if (cid == 0 || count[cid] == 1) {        // изолирован
            // cv::compare быстрее, чем == в выражении
            cv::compare(comp, cid, tmpMask, cv::CMP_EQ);
            zones.push_back( ZoneMask{ lbl, tmpMask.clone() } );
        }
    return zones;
}
