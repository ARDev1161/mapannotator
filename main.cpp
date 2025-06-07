#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include "yaml-cpp/yaml.h"

#include "mapgraph/zonegraph.hpp"
#include "mapgraph/zone_graph_dot.hpp"
#include "mapgraph/zone_graph_draw.hpp"
#include "preparing/mappreprocessing.hpp"
#include "segmentation/segmentation.hpp"
#include "segmentation/labelmapping.hpp"
#include "segmentation/endpoints.h"
#include "segmentation/zoneclassifier.h"
#include "config.hpp"
#include "utils.hpp"

using namespace mapping;

/**
 * Создаёт таблицу смежности зон по одному изображению меток.
 *
 * @param zones             CV_32S: 0 = стены / фон, >0 = номер зоны
 * @param use8Connectivity  true  → 8-связность (считаем диагонали),
 *                           false → 4-связность (только ↑↓←→)
 * @return unordered_map<label, vector<label-соседей>>
 */
std::unordered_map<int, std::vector<int>>
buildZoneAdjacency(const cv::Mat1i& zones,
                   bool use8Connectivity = false)
{
    CV_Assert(!zones.empty() && zones.type() == CV_32S);

    const cv::Size sz = zones.size();

    const std::vector<cv::Point> off4{{1,0},{0,1}};
    const std::vector<cv::Point> off8{{1,0},{0,1},{1,1},{1,-1}};
    const auto& offs = use8Connectivity ? off8 : off4;

    std::unordered_map<int, std::unordered_set<int>> tmpAdj;

    for (int y = 0; y < sz.height; ++y)
    {
        const int* row = zones.ptr<int>(y);
        for (int x = 0; x < sz.width; ++x)
        {
            int lbl = row[x];
            if (lbl <= 0) continue;                     // фон

            for (const auto& d : offs)
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

    for (auto& [lbl, set] : tmpAdj)
    {
        adjacency[lbl] = {set.begin(), set.end()};
        std::sort(adjacency[lbl].begin(), adjacency[lbl].end());
    }
    return adjacency;
}

/** Строит индекс: label → указатель на mask */
inline std::unordered_map<int, const cv::Mat1b*>
buildZoneIndex(const std::vector<ZoneMask>& zones)
{
    std::unordered_map<int, const cv::Mat1b*> index;
    index.reserve(zones.size());

    for (const auto& z : zones)
        index.emplace(z.label, &z.mask);

    return index;
}

void annotateGraph(ZoneGraph& graph, const std::string& rulefile)
{
    static ZoneClassifier clf{rulefile};

    for (auto& node : graph.allNodes())        // итератор по узлам
    {
        ZoneFeatures f = node->features();  // ваш метод получения признаков
        ZoneType     t = clf.classify(f);

        node->setType(t);                   // сохраните / подпишите узел
        std::cout << "Zone #" << node->id()
                  << " → " << t.path() << '\n';
    }
}

void buildGraph(ZoneGraph& graphOut, std::vector<ZoneMask> zones, cv::Mat1i zonesMat, std::unordered_map<int, cv::Point> centroids) {

    std::unordered_map<ZoneType, NodePtr> node;   // для удобства соединений

    auto add = [&](ZoneId id, ZoneType t, cv::Point2d c, double area, double perimeter)
    {
        node[t] = graphOut.addNode(id, t, area, perimeter, c,
                            {{{c.x-0.5,c.y-0.5},{c.x+0.5,c.y-0.5},
                              {c.x+0.5,c.y+0.5},{c.x-0.5,c.y+0.5}}});
    };

    auto W = [](double m){ return m; };        // удобный alias

    std::unordered_map<int, std::vector<int>> zoneGraph =
            buildZoneAdjacency(zonesMat, /*use8Connectivity=*/true); // Таблица смежности графа

    // Assume map resolution is 5 cm per pixel ⇒ area per pixel = 0.05 × 0.05 m².
    constexpr double kAreaPerPixel = 0.05 * 0.05;

    auto zoneIndex = buildZoneIndex(zones);

    // Создаем ноды графа
    for(auto centroid: centroids){

        // размещаем вершины (координаты в «метрах»)
        add(centroid.first,
            ZoneType{},
            pixelToWorld(centroid.second, mapInfo),
            computeWhiteArea(zoneIndex[centroid.first], kAreaPerPixel),
                10);

        // строим рёбра
        std::cerr << "Zone " << std::to_string(centroid.first) << centroid.second << " neighbours:";
        for(auto neighbour: zoneGraph[centroid.first]){
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

/**
 * @param  seg        CV_32S: 0 = стены/фон, ≥1 = код зоны
 * @param  winName    название окна для cv::imshow ("" → не показывать)
 * @param  pngPath    куда сохранить PNG ("" → не сохранять)
 * @param  colormap   см. COLORMAP_* (например, COLORMAP_JET, HOT, VIRIDIS …)
 */
cv::Mat3b colorizeSegmentation(const cv::Mat1i& seg,
                               const cv::Mat1b& wallMask,
                               const std::string& winName  = "Segmentation",
                               std::string      pngPath  = "segmentation.png",
                               int              colormap = cv::COLORMAP_JET)
{
    CV_Assert(!seg.empty() && seg.type() == CV_32S);

    /* 1.  нормируем [0 .. maxLabel] → [0 .. 255]  ------------------------- */
    double minVal, maxVal;
    cv::minMaxLoc(seg, &minVal, &maxVal);
    double scale = (maxVal > 0) ? 255.0 / maxVal : 1.0;

    cv::Mat1b seg8u;
    seg.convertTo(seg8u, CV_8U, scale);       // CV_8U для applyColorMap

    /* 2.  применяем цветовую карту  --------------------------------------- */
    cv::Mat3b color;
//    cv::Mat1i hashed = (seg * 1315423911u) & 0xFF;   // простое хеш-перемешивание
//    hashed.convertTo(seg8u, CV_8U);
    cv::applyColorMap(seg8u, color, colormap);

    /* 3.  делаем фон серым для контраста (опц.) -------------------- */
    // фон = те же пиксели, что были 0 в seg (т.е. seg8u == 0)
    color.setTo(cv::Vec3b(50,50,50), seg8u == 0);

    // добавляем стены
    double alpha = 0.9;                  // непрозрачность 0…1 (0.6 = 60 %)
    cv::Scalar wallColor(0, 0, 0);     // чёрные стены (BGR)

    /* создаём BGR-картинку только для стен */
    cv::Mat overlay(color.size(), color.type(), wallColor);

    /* смешиваем: dst = imgColor*(1-α) + overlay*α  только там, где стена */
    cv::Mat blended = color.clone();
    overlay.copyTo(blended, wallMask);   // overlay → blended (там, где стена)

    /* linearBlend = img*(1-α) + blended*α, но только по маске */
    cv::addWeighted(color, 1.0, blended, alpha, 0.0, blended);


    /* 4.  вывод / сохранение --------------------------------------------- */
    if (!winName.empty()) {
        showMat(winName, blended);
        cv::waitKey(0);
    }
    if (!pngPath.empty())
        cv::imwrite(pngPath, blended);


    return blended;                              // на случай дальнейшей работы
}

/**
 * @brief Построить маску области, достижимой из заданного центроида по свободным
 *        пикселям (значение 0) с помощью Flood Fill.
 *
 * @param freeMap    Одноканальная карта (CV_8UC1), где 0 — свободная ячейка,
 *                   любой ненулевой байт — препятствие/стена.
 * @param seed       Точка-центроид, откуда начинается заливка.
 * @param connectivity 4 или 8 (по умолчанию 8).
 * @return cv::Mat1b того же размера, где 255 помечает залитую область,
 *         0 — остальное.
 *
 * Примечания
 * ----------
 * 1.  Исходная карта **не** изменяется (используется FLOODFILL_MASK_ONLY).
 * 2.  Маска floodFill требует рамку в 1 пиксель; она удаляется перед возвратом.
 * 3.  Если seed лежит на препятствии, функция вернёт нулевую маску.
 */
cv::Mat1b buildFloodMask(const cv::Mat1b& freeMap,
                         const cv::Point& seed,
                         int connectivity = 8)
{
    CV_Assert(freeMap.type() == CV_8UC1);
    CV_Assert(seed.x >= 0 && seed.x < freeMap.cols &&
              seed.y >= 0 && seed.y < freeMap.rows);

    // Если в точке-семени нет свободного пикселя ─ нечего заливать
    if (freeMap(seed) != 0)
        return cv::Mat1b::zeros(freeMap.size());

    // Маска floodFill должна быть на 2 px больше по каждому измерению
    cv::Mat1b mask(freeMap.rows + 2, freeMap.cols + 2, uchar(0));

    // Выполняем заливку только в маске
    const int flags = connectivity | cv::FLOODFILL_MASK_ONLY | (255 << 8);
    cv::floodFill(freeMap, mask, seed,                 /* newVal */ 0,
                  /* rect */         nullptr,
                  /* loDiff/upDiff*/ cv::Scalar(), cv::Scalar(),
                  flags);

    // Обрезаем рамку и возвращаем
    return mask(cv::Rect(1, 1, freeMap.cols, freeMap.rows)).clone();
}

//--------------------------------------------------------------------------
//  mergeLonelyFreeAreas()
//--------------------------------------------------------------------------
/// Находит свободные компоненты, соприкасающиеся ровно с одной зоной, –
/// расширяет её и «закрашивает» occ; возвращает, сколько таких компонентов
/// обработано.
///
/// Конвенция (можно адаптировать):
///   * occ        – 0  = свободно, 255 = занято (зона / препятствие)
///   * background – 0  = стенa,    255 = пространство, куда робот может ехать
///   * zone.mask  – 255 = принадлежит зоне
///
int mergeLonelyFreeAreas(cv::Mat1b&                 occ,        // [in/out]
                         const cv::Mat1b&           background, // [in]
                         std::vector<ZoneMask>&     allZones)   // [in/out]
{
    CV_Assert( occ.size() == background.size() &&
               occ.type() == CV_8UC1 && background.type() == CV_8UC1 );

    //--- 1. Карта меток (labelMap): >0 внутри зон, 0 – везде ещё ----------
    cv::Mat1i labelMap = cv::Mat1i::zeros(occ.size());
    std::unordered_map<int, std::size_t> label2idx;  // label → index в allZones

    for (std::size_t i = 0; i < allZones.size(); ++i)
    {
        const auto& z = allZones[i];
        CV_Assert(z.mask.size() == occ.size() && z.mask.type() == CV_8UC1);
        labelMap.setTo(z.label, z.mask);             // заполняем числами меток
        label2idx[z.label] = i;
    }

    //--- 2. Выделяем свободные области (0 в occ И 255 в background) -------
    cv::Mat1b freeMask;
    cv::bitwise_and(occ == 0, background, freeMask);   // чётко бинарно

    cv::Mat1i compLabels;
    int nComp = cv::connectedComponents(freeMask, compLabels, 8, CV_32S);

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,{3,3});
    int mergedCount = 0;

    //--- 3. Проходим по всем компонентам, кроме фона #0 -------------------
    for (int id = 1; id < nComp; ++id)
    {
        cv::Mat1b regionMask = (compLabels == id);     // 255 внутри компоненты

        // 3a. Расширяем на 1 пиксель, чтобы «пощупать» соседей
        cv::Mat1b dilated;
        cv::dilate(regionMask, dilated, kernel, {-1,-1}, 1);

        // 3b. Собираем уникальные метки, с которыми есть контакт
        std::vector<cv::Point> pts;
        cv::findNonZero(dilated, pts);

        std::unordered_set<int> adjLabels;
        for (const auto& p : pts)
        {
            int lbl = labelMap.at<int>(p);
            if (lbl > 0) adjLabels.insert(lbl);
        }

        if (adjLabels.size() == 1)                      // ровно одна зона!
        {
            int lbl = *adjLabels.begin();
            auto it = label2idx.find(lbl);
            if (it == label2idx.end()) continue;        // на всякий (не должно)

            ZoneMask& z = allZones[it->second];

            // 3c. Добавляем компоненту к зоне
            z.mask.setTo(255, regionMask);              // зона расширилась
            occ.setTo(255, regionMask);                 // занято → 255

            ++mergedCount;
        }
    }
    return mergedCount;
}

/*-------------------------------------------------------------------------*/
/*  eraseWallsFromZones                                                    */
/*-------------------------------------------------------------------------*/
/**
 * @brief   Убирает «наплывы» масок на стены.
 *
 * @param   background  CV_8UC1: 0 – стены, 255 – свободное пространство
 * @param   allZones    вектор зон; их mask модифицируются на месте
 *
 * Алгоритм:  z.mask &= background   (битовое И);
 * после чего остаётся только то, что лежит на доступной области пола.
 */
void eraseWallsFromZones(const cv::Mat1b&       background,
                         std::vector<ZoneMask>& allZones)
{
    CV_Assert(!background.empty() && background.type() == CV_8UC1);

    for (auto& z : allZones)
    {
        CV_Assert(z.mask.size() == background.size() &&
                  z.mask.type()  == CV_8UC1);

        // 1. Убираем стены: 255 & 0 → 0, 255 & 255 → 255
        cv::bitwise_and(z.mask, background, z.mask);

        // 2. Гарантируем чёткую бинарность (на случай !=0 артефактов)
        cv::threshold(z.mask, z.mask, 0, 255, cv::THRESH_BINARY);
    }
}

/*-------------------------------------------------------------------------*/
/*  keepCentroidComponent                                                  */
/*-------------------------------------------------------------------------*/
/**
 * @brief   В каждой зоне оставляет только ту связную компоненту,
 *          которая содержит переданный центроид.  Все остальные
 *          части стираются.  Если передан указатель на карту занятости
 *          `occ`, удалённые пиксели в ней помечаются как свободные (0).
 *
 * @param   centroids  label → cv::Point   (координаты центроидов)
 * @param   allZones   вектор зон; их mask модифицируются на месте
 * @param   occ        [опц.] карта занятости. 0 = свободно, 255 = занято.
 *                    Должна быть того же размера и CV_8UC1.
 *
 * Условия: mask’и бинарные (0/255) или приводим к таковым.
 */
void keepCentroidComponent(const std::unordered_map<int, cv::Point>& centroids,
                           std::vector<ZoneMask>&                     allZones,
                           cv::Mat1b*                                occ = nullptr)
{
    // если occ указан – проверяем корректность один раз
    if (occ) {
        CV_Assert(!occ->empty() && occ->type() == CV_8UC1);
    }

    for (auto& z : allZones)
    {
        auto it = centroids.find(z.label);
        if (it == centroids.end()) {
            std::cerr << "[keepCentroidComponent] no centroid for label "
                      << z.label << '\n';
            continue;
        }

        const cv::Point& c = it->second;
        cv::Mat1b& m       = z.mask;

        CV_Assert(!m.empty() && m.type() == CV_8UC1);

        // --- проверяем, что точка в пределах матрицы --------------------
        if (c.x < 0 || c.x >= m.cols || c.y < 0 || c.y >= m.rows) {
            std::cerr << "[keepCentroidComponent] centroid outside image for label "
                      << z.label << '\n';
            continue;
        }

        // --- маска должна быть строго 0/255 -----------------------------
        cv::threshold(m, m, 0, 255, cv::THRESH_BINARY);

        if (m.at<uchar>(c) == 0) {          // центроид лежит не в зоне
            std::cerr << "[keepCentroidComponent] centroid not inside mask for label "
                      << z.label << '\n';
            continue;
        }

        // --- помечаем компоненты ----------------------------------------
        cv::Mat1i labels;
        cv::connectedComponents(m, labels, 8, CV_32S);

        int compLbl = labels.at<int>(c);    // id нужной компоненты (>0)

        // новая маска = только нужная CC
        cv::Mat1b newMask;
        cv::compare(labels, compLbl, newMask, cv::CMP_EQ);   // 255 в нужной CC

        // --- если карта занятости передана, обнуляем удалённые пиксели ---
        if (occ) {
            cv::Mat1b removed;
            cv::bitwise_and(m, ~newMask, removed);  // то, что было 255 и стало 0
            occ->setTo(0, removed);                 // 0 = свободно
        }

        m = newMask;                        // заменяем старую маску
    }
}

/**
 *  attachPixelsToNearestZone
 *  -------------------------
 *  • occupancy   — CV_8UC1: 0 = свободно (чёрное), 255 = занято
 *  • allZones    — список масок (255 = зона, 0 = фон) того же размера
 *
 *  Для каждого свободного пикселя находит ближайшую зону (по евклидовой
 *  дистанции), добавляет пиксель в маску этой зоны и ставит 255 в occupancy.
 *
 *  @return сколько пикселей было присвоено
 *
 *  Сложность O(N_pix · N_zones), но с предрассчитанными картами расстояний
 *  (cv::distanceTransform) работает заметно быстрее, чем прямой перебор
 *  всех точек каждой зоны.
 */
std::size_t attachPixelsToNearestZone(cv::Mat1b&              occupancy,
                                      std::vector<ZoneMask>&  allZones)
{
    CV_Assert(!occupancy.empty() && occupancy.type() == CV_8UC1);

    const int rows = occupancy.rows, cols = occupancy.cols;
    const int Z = static_cast<int>(allZones.size());

    /* ── 1. Расстояние до каждой зоны (предрасчёт) ─────────────────── */
    std::vector<cv::Mat1f> distMaps(Z);            // float32

    for (int zi = 0; zi < Z; ++zi)
    {
        auto& z = allZones[zi];
        CV_Assert(z.mask.size() == occupancy.size() && z.mask.type() == CV_8UC1);

        cv::Mat1b invMask;
        cv::bitwise_not(z.mask, invMask);          // зона → 0, остальное → 255

        cv::distanceTransform(invMask,
                              distMaps[zi],
                              cv::DIST_L2,
                              3);                  // 3×3 маска, L2-норма
    }

    /* ── 2. Проходим по свободным пикселям и ищем ближайшую зону ───── */
    std::size_t attached = 0;

    for (int y = 0; y < rows; ++y)
    {
        uchar* occRow = occupancy.ptr<uchar>(y);

        for (int x = 0; x < cols; ++x)
        {
            if (occRow[x] != 0)                    // уже занято
                continue;

            /* — ищем минимальное расстояние — */
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

            /* — присваиваем пиксель найденной зоне — */
            if (bestIdx >= 0)
            {
                allZones[bestIdx].mask(y, x) = 255;  // добавить в зону
                occRow[x]                    = 255;  // теперь занято
                ++attached;
            }
        }
    }

    return attached;                                // кол-во обработанных пикселей
}

std::vector<ZoneMask>
segmentByGaussianThreshold(const cv::Mat1b& srcBinary,   // 0 = стены, 255 = пусто
                           LabelsInfo& labelsOut,
                           int   maxIter      = 40,
                           double sigmaStep   = 0.25,     // приращение σ на итерацию
                           double    threshold   = 0.5)     // T для THRESH_BINARY
{
    CV_Assert(!srcBinary.empty() && srcBinary.type() == CV_8UC1);

    /* 0.  центроиды свободных областей исходной карты --------------------- */
    labelsOut = LabelMapping::computeLabels(srcBinary, /*invert=*/false);

    std::unordered_map<int, cv::Point> todo = labelsOut.centroids; // ещё не изолированы
    std::vector<ZoneMask> allZones;                           // результат

    cv::Mat blurred, bin;

    cv::Mat1b eroded = srcBinary.clone();

    cv::Mat1b src255;
    srcBinary.convertTo(src255, CV_8U, 255);

    for (int iter = 0; iter < maxIter && !todo.empty(); ++iter)
    {
        double sigma = (iter + 1) * sigmaStep;               // растёт σ

        /* 0.  подготовка маски стен с усиленной маской в endpoints --------------------------- */
        // Расширяем стены
        cv::Mat1b kernel = cv::getStructuringElement(
                               cv::MORPH_RECT, cv::Size(2*1+1, 2*1+1));
        cv::erode(eroded, eroded, kernel, cv::Point(-1,-1), 1);

        /* 1.  размытие исходной бинарной карты --------------------------- */
        cv::GaussianBlur(eroded, blurred, cv::Size(0,0), sigma, sigma,
                         cv::BORDER_REPLICATE);

        /* 2.  порог → новая бинарная карта «виртуально утолщённых» стен -- */
        cv::threshold(blurred, bin, threshold, 1, cv::THRESH_BINARY);

        cv::Mat1b seg8u;
        bin.convertTo(seg8u, CV_8U, 255);       // CV_8U для applyColorMap
//        showMat("blurred threshold " + std::to_string(iter), seg8u);

        if (cv::countNonZero(bin) == 0) {        // всё стало стенами
            std::cerr << "[warn] free space vanished at iter " << iter << '\n';
            break;
        }

        /* 3.  ищем компоненты, изолированы ли центроиды ------------------- */
        auto zones = LabelMapping::extractIsolatedZones(bin, todo, /*invertFree=*/true);

        int it = 0;
        for (auto& z : zones) {
            cv::dilate(z.mask, z.mask, kernel, cv::Point(-1,-1), iter + 1); // +1 due to eroding into main
            allZones.push_back( std::move(z) );
            todo.erase(z.label);   // метка выполнена
        }
    }

    // Обработка необработанных меток
    if (!todo.empty())
    {
        // Building occupancy mask
        cv::Mat1b occ = LabelMapping::buildOccupancyMask(src255, allZones);

        // Проходим по меткам которые остались не изолированными
        for(auto it = todo.begin(); it != todo.end();)
        {
            // проверяем если оставшиеся метки на фоне(под ними нет никакой зоны)
            // такие зоны появляются если на концах нет выступов и их просто сжимает(например узкий коридор)
            if(occ(it->second) == 0)
            {
                std::cerr << "Not segmented (no zone under label): "
                          << it->first << it->second << '\n';

                // flood fill
                ZoneMask z;
                z.label = it->first;
                z.mask  = buildFloodMask(occ, it->second);

                // Add zone to occupancy mask
                checkMatCompatibility(occ, z.mask);
                occ.setTo(255, z.mask);

                allZones.push_back( std::move(z) );

                it = todo.erase(it);           // erase возвращает следующий корректный итератор
            }
            else
                ++it;                           // идём дальше только если не стирали

        }

        eraseWallsFromZones(srcBinary, allZones);

        std::size_t n = attachPixelsToNearestZone(occ, allZones);

        // после вычитания стен могут появиться отделенные части зон с обеих сторон стены.
        keepCentroidComponent(labelsOut.centroids, allZones, &occ); // Оставляем только части зон где лежит центроид
        int added = mergeLonelyFreeAreas(occ, srcBinary, allZones);
        std::cout << "Добавлено участков: " << added << '\n';


//        showMat("occ ", occ);
        std::cerr << "[warn] maxIter reached, "
                  << todo.size() << " zones still not isolated: ";

        for(auto zone: todo)
            std::cerr << zone.first << zone.second << " ";
        std::cerr << std::endl;
    }

    return allZones;                             // набор масок зон
}


int main(int argc, char** argv)
{
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <map.pgm> [config.yaml]" << std::endl;
        return 1;
    }

    SegmenterConfig segmenterConfig;

    std::string pgmFile = argv[1];
    // Если передан файл конфигурации, используем его, иначе "default.yml"
    std::string configFile = (argc >= 3) ? argv[2] : "default.yml";

    // Загружаем YAML-конфигурацию
    YAML::Node config;
    try {
        config = YAML::LoadFile(configFile);
    } catch(const std::exception &e) {
        std::cerr << "Failed to load config file: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Loaded config from: " << configFile << std::endl;

    // Определяем имя map.yaml: то же, что и pgmFile, только с расширением .yaml
    std::string mapYamlFile = pgmFile.substr(0, pgmFile.find_last_of('.')) + ".yaml";
    bool mapYamlLoaded = false;
    std::vector<double> map_origin; // ожидается вектор из 3 элементов: [x, y, theta]
    double resolution = 1.0;         // значение по умолчанию (метров на пиксель)

    if (std::filesystem::exists(mapYamlFile)) {
        try {
            YAML::Node map_yaml = YAML::LoadFile(mapYamlFile);
            std::cout << "Loaded map YAML from: " << mapYamlFile << std::endl;
            if (map_yaml["origin"]) {
                map_origin = map_yaml["origin"].as<std::vector<double>>();
                if(map_origin.size() < 3) {
                    std::cerr << "Map YAML 'origin' field has insufficient elements (expected at least 3)." << std::endl;
                } else {
                    std::cout << "Map origin: x = " << map_origin[0]
                              << ", y = " << map_origin[1]
                              << ", theta = " << map_origin[2] << std::endl;
                }
            } else {
                std::cout << "No origin information in map YAML; origin remains unchanged." << std::endl;
            }
            if (map_yaml["resolution"]) {
                resolution = map_yaml["resolution"].as<double>();
                std::cout << "Map resolution: " << resolution << " meters/pixel" << std::endl;
            } else {
                std::cout << "No resolution information in map YAML; using default: " << resolution << std::endl;
            }
            mapYamlLoaded = true;
        } catch (const std::exception &e) {
            std::cerr << "Error loading map YAML: " << e.what() << std::endl;
        }
    } else {
        std::cout << "Map YAML file not found (" << mapYamlFile << "); origin remains unchanged." << std::endl;
    }

    // Если не удалось загрузить map.yaml, используем значения по умолчанию для origin
    if(!mapYamlLoaded || map_origin.size() < 3) {
        map_origin = {0.0, 0.0, 0.0};
    }

    // Загружаем карту (PGM) в оттенках серого
    cv::Mat raw = cv::imread(pgmFile, cv::IMREAD_GRAYSCALE);
    if(raw.empty()){
        std::cerr << "Failed to load image from: " << pgmFile << std::endl;
        return 1;
    }
    std::cout << "Loaded image: " << pgmFile << " (size: " << raw.cols << "x" << raw.rows << ")" << std::endl;

//    showMat("Raw Map", raw);

    // Заполнение структуры MapInfo
    mapInfo.resolution = resolution;
    mapInfo.originX = map_origin[0];
    mapInfo.originY = map_origin[1];
    mapInfo.theta = map_origin[2];
    mapInfo.height = raw.rows;
    mapInfo.width = raw.cols;

    cv::Mat raw8u;
    raw.convertTo(raw8u, CV_8UC1);

    cv::Mat aligned;
    MapPreprocessing::mapAlign(raw8u, aligned, segmenterConfig.alignmentConfig);

//    showMat("aligned Map", aligned);

    // Этап денойзинга.
    auto [rank, cropInfo] = MapPreprocessing::generateDenoisedAlone(aligned, segmenterConfig.denoiseConfig);

//    showMat("Denoised Map", rank);

    // Расширенние черных зон(препятствия)
    cv::Mat1b binaryDilated = erodeBinary(rank, segmenterConfig.dilateConfig.kernelSize, segmenterConfig.dilateConfig.iterations);


    cv::Mat1b wallMask;
    cv::compare(binaryDilated, 0, wallMask, cv::CMP_EQ);   // 255 там, где стена
//    cv::Mat1b skeleton;
//    cv::Mat sk8;
//    cv::ximgproc::thinning(wallMask, skeleton, cv::ximgproc::THINNING_ZHANGSUEN);
//    skeleton.convertTo(sk8, CV_8U, 255);
//            std::cout << "skeleton non-zero = " << cv::countNonZero(skeleton) << std::endl;
//    showMat("Skeleton", sk8);

//    std::vector<cv::Point> ends;
//    showMat("Skeleton + endpoints", visualizeSkeletonEndpoints(skeleton, &ends));

//    cv::Mat1b endpointsMask = drawSkeletonEndpoints(skeleton, 3);
//    showMat("Endpoints mask", endpointsMask); // inflation radius

    LabelsInfo labels;
    auto zones = segmentByGaussianThreshold(binaryDilated, labels, 50, 0.5);

    cv::Mat1i segmentation = cv::Mat::zeros(binaryDilated.size(), CV_32S);
            int iter = 0;
    for (const auto& z : zones) {

        cv::Mat1b seg8u;
        z.mask.convertTo(seg8u, CV_8U, 255);       // CV_8U для applyColorMap
        showMat("Zone №" + std::to_string(z.label), seg8u);

        segmentation.setTo(z.label, z.mask);
    }

    cv::Mat1i  seg = segmentation;                     // ваша сегментация (0 = стены)
    double minVal, maxVal;
    cv::minMaxLoc(seg, &minVal, &maxVal);     // нужен maxVal ≥ 1
    cv::Mat1b gray8;
    seg.convertTo(gray8, CV_8U,
                  maxVal ? 255.0 / maxVal : 1.0,   // α (scale)
                  0);                              // β (shift)

    cv::imshow("Segmentation", gray8);

    ZoneGraph graph;
    buildGraph(graph, zones, segmentation, labels.centroids);

    cv::Mat3b vis = colorizeSegmentation(segmentation, wallMask,
                    "Rooms colored",
                    "rooms.png",
                    cv::COLORMAP_JET);

//    exampleZoneGraph();

    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);
}
