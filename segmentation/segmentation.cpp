#include "segmentation.hpp"
#include <stdexcept>
#include <algorithm>

// Находит ограничивающий прямоугольник (bounding box) для карты.
// Используется первый пиксель как фон, далее определяется x1, x2, y1 и y2.
cv::Rect Segmentation::findBB(const cv::Mat& map) {
    cv::Mat map_uint8;
    if (map.type() == CV_32F || map.type() == CV_32FC1)
        map.convertTo(map_uint8, CV_8U, 255.0);
    else
        map_uint8 = map.clone();

    int width = map_uint8.cols;
    int height = map_uint8.rows;
    int padding = 1;
    uchar background = map_uint8.at<uchar>(0, 0);
    int x1 = 0;
    for (int x = 0; x < width; x++) {
        bool allBg = true;
        for (int y = 0; y < height; y++) {
            if (map_uint8.at<uchar>(y, x) != background) {
                allBg = false;
                break;
            }
        }
        if (!allBg) {
            x1 = std::max(x - padding, 0);
            break;
        }
    }
    int x2 = width;
    for (int x = width - 1; x >= 0; x--) {
        bool allBg = true;
        for (int y = 0; y < height; y++) {
            if (map_uint8.at<uchar>(y, x) != background) {
                allBg = false;
                break;
            }
        }
        if (!allBg) {
            x2 = std::min(x + padding, width);
            break;
        }
    }
    int y1 = 0;
    for (int y = 0; y < height; y++) {
        bool allBg = true;
        for (int x = 0; x < width; x++) {
            if (map_uint8.at<uchar>(y, x) != background) {
                allBg = false;
                break;
            }
        }
        if (!allBg) {
            y1 = std::max(y - padding, 0);
            break;
        }
    }
    int y2 = height;
    for (int y = height - 1; y >= 0; y--) {
        bool allBg = true;
        for (int x = 0; x < width; x++) {
            if (map_uint8.at<uchar>(y, x) != background) {
                allBg = false;
                break;
            }
        }
        if (!allBg) {
            y2 = std::min(y + padding, height);
            break;
        }
    }
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

// Вычисляет информацию для единичного кадрирования.
Segmentation::CropInfo Segmentation::cropSingleInfo(const cv::Mat& map, int padding)
{
    /* ---------- sanity-checks ------------------------------------------ */
    if (map.empty())
        throw std::invalid_argument("cropSingleInfo(): map is empty");

    if (padding < 0)
        throw std::invalid_argument("cropSingleInfo(): padding must be ≥ 0");

    CV_Assert(map.type() == CV_8UC1 && "expecting single-channel map");

    /* ---------- bounding box of non-zero pixels ------------------------ */
    const cv::Rect bb = findBB(map);             // may return {0,0,0,0}
    if (bb.width == 0 || bb.height == 0)
        return {};                               // nothing to crop

    const int width = map.cols;
    const int height = map.rows;

    /* ограничиваем padding, чтобы не выйти за границы */
    const int maxPad = std::min({ bb.x,
                                  bb.y,
                                  width  - bb.x - bb.width,
                                  height - bb.y - bb.height });
    padding = std::min(padding, maxPad);

    CropInfo info;
    info.left   = bb.x - padding;                           // ≥ 0
    info.top    = bb.y - padding;                           // ≥ 0
    info.right  = width  - (bb.x + bb.width)  - padding;    // ≥ 0
    info.bottom = height - (bb.y + bb.height) - padding;    // ≥ 0
    return info;
}

// Вычисляет информацию для пакетного кадрирования двух карт.
Segmentation::CropInfo Segmentation::cropBundleInfo(const cv::Mat& rankMap, const cv::Mat& trackMap) {
    if (rankMap.size() != trackMap.size()) {
        throw std::runtime_error("The two maps must be the same size.");
    }
    int width = rankMap.cols;
    int height = rankMap.rows;
    cv::Rect r1 = findBB(rankMap);
    cv::Rect r2 = findBB(trackMap);
    int x1 = r1.x;
    int y1 = r1.y;
    int x2 = r1.x + r1.width;
    int y2 = r1.y + r1.height;
    int a1 = r2.x;
    int b1 = r2.y;
    int a2 = r2.x + r2.width;
    int b2 = r2.y + r2.height;
    CropInfo info;
    info.left = std::min(x1, a1);
    info.top = std::min(y1, b1);
    info.right = width - std::max(x2, a2);
    info.bottom = height - std::max(y2, b2);
    return info;
}

// Кадрирование изображения по заданной информации.
cv::Mat Segmentation::cropBackground(const cv::Mat& map, const CropInfo& info) {
    int newWidth = map.cols - info.left - info.right;
    int newHeight = map.rows - info.top - info.bottom;
    cv::Rect roi(info.left, info.top, newWidth, newHeight);
    return map(roi).clone();
}

// Восстанавливает исходный размер изображения, добавляя фон.
cv::Mat Segmentation::uncropBackground(const cv::Mat& map, const CropInfo& info, const cv::Scalar& background) {
    int newWidth = info.left + map.cols + info.right;
    int newHeight = info.top + map.rows + info.bottom;
    cv::Mat uncropped(newHeight, newWidth, map.type(), background);
    cv::Rect roi(info.left, info.top, map.cols, map.rows);
    map.copyTo(uncropped(roi));
    return uncropped;
}

// Заменяет пиксели с заданным ярлыком границы на 255.
// Если border_label == -1, определяется максимальное значение в изображении.
cv::Mat Segmentation::remapBorder(const cv::Mat& map, int border_label) {
    cv::Mat res = map.clone();
    if (border_label == -1) {
        double minVal, maxVal;
        cv::minMaxLoc(map, &minVal, &maxVal);
        border_label = static_cast<int>(maxVal);
    }
    for (int y = 0; y < res.rows; y++) {
        for (int x = 0; x < res.cols; x++) {
            if (res.at<uchar>(y, x) == border_label)
                res.at<uchar>(y, x) = 255;
        }
    }
    return res;
}

// Удаляет небольшие связные компоненты из бинарного изображения.
cv::Mat Segmentation::removeSmallConnectedComponents(const cv::Mat& src, const std::string& method,
                                                       int min_size, int connectivity, bool debug) {
    // Приводим исходное изображение к 8U, если оно не имеет этот тип.
    cv::Mat src_uint8;
    if (src.type() != CV_8U)
        src.convertTo(src_uint8, CV_8U, 255.0);
    else
        src_uint8 = src;

    // Находим связанные компоненты
    cv::Mat labels, stats, centroids;
    int num_components = cv::connectedComponentsWithStats(src_uint8, labels, stats, centroids, connectivity, CV_32S);

    // Собираем размеры найденных компонентов (пропуская фон с меткой 0)
    std::vector<int> sizes;
    for (int i = 1; i < num_components; i++) {
        sizes.push_back(stats.at<int>(i, cv::CC_STAT_AREA));
    }

    // Определяем порог фильтрации компонентов по выбранному методу
    double threshold_val = 0.0;
    if (method == "fixed") {
        threshold_val = min_size;
    } else if (method == "mean") {
        double sum = 0;
        for (int s : sizes) sum += s;
        threshold_val = sum / sizes.size();
    } else if (method == "median") {
        std::vector<int> sorted = sizes;
        std::sort(sorted.begin(), sorted.end());
        threshold_val = sorted[sorted.size() / 2];
    } else {
        throw std::runtime_error("Unknown method in removeSmallConnectedComponents");
    }

    // Создаем бинарную маску результата в типе 8U (0 или 255)
    cv::Mat result_mask = cv::Mat::zeros(src_uint8.size(), CV_8U);
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            int label = labels.at<int>(y, x);
            if (label > 0) { // пропускаем фон
                int blob_index = label - 1;
                if (sizes[blob_index] >= threshold_val) {
                    result_mask.at<uchar>(y, x) = 255;
                }
            }
        }
    }

    // Определяем, какое значение будем считать "истинным" в результирующем изображении,
    // в зависимости от типа исходного изображения.
    double trueValue = 1.0;
    int depth = src.depth();
    switch (depth) {
        case CV_8U:  trueValue = 255; break;
        case CV_8S:  trueValue = 1;   break;
        case CV_16U: trueValue = 65535; break;
        case CV_16S: trueValue = 1;   break;
        case CV_32S: trueValue = 1;   break;
        case CV_32F: trueValue = 1.0; break;
        case CV_64F: trueValue = 1.0; break;
        default:     trueValue = 1.0; break;
    }

    // Приводим бинарную маску (0/255) к типу исходного изображения.
    // Для этого масштабируем значение 255 до trueValue.
    cv::Mat result;
    result_mask.convertTo(result, src.type(), trueValue / 255.0);

    return result;
}


// Генерация меток (seeds) с использованием гауссового размытия и последующей бинаризации.
cv::Mat Segmentation::generateGaussianSeeds(const cv::Mat& src, const cv::Mat& background_mask,
                                              double sigma, double threshold) {
    cv::Mat blurred;
    int ksize = std::max(3, static_cast<int>(sigma * 6 + 1));
    if (ksize % 2 == 0) ksize++;
    cv::GaussianBlur(src, blurred, cv::Size(ksize, ksize), sigma);
    cv::Mat binary = makeBinary(blurred, threshold);
    cv::Mat binary_uint8;
    binary.convertTo(binary_uint8, CV_8U, 255.0);
    cv::Mat inv = makeInvert(binary_uint8);
    cv::Mat labels;
    cv::connectedComponents(inv, labels, 8, CV_32S);
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            if (background_mask.at<uchar>(y, x) != 0) {
                labels.at<int>(y, x) = 255;
            }
        }
    }
    return inv;
}

// Сегментация методом watershed.
cv::Mat Segmentation::createWatershedSegment(const cv::Mat& src, const cv::Mat& labels, int connectivity) {
    cv::Mat src8;
    // Если исходное изображение не типа CV_8U, конвертируем его (предполагается, что src в диапазоне [0,1])
    if (src.type() != CV_8U) {
        src.convertTo(src8, CV_8U, 255.0);
    } else {
        src8 = src;
    }

    // Преобразуем в цветное изображение (CV_8UC3)
    cv::Mat src3;
    if (src8.channels() != 3)
        cv::cvtColor(src8, src3, cv::COLOR_GRAY2BGR);
    else
        src3 = src8.clone();

    // Преобразуем labels в матрицу типа CV_32S для watershed.
    cv::Mat markers;
    labels.convertTo(markers, CV_32S);

    // Проверяем, что markers не пуст и имеет правильный тип.
    CV_Assert(!markers.empty());
    CV_Assert(markers.type() == CV_32S);

    // Убедимся, что markers непрерывна.
    if (!markers.isContinuous())
        markers = markers.clone();

    // Если размер markers не совпадает с размером src3, изменим его размер.
    if (markers.size() != src3.size()) {
        cv::resize(markers, markers, src3.size(), 0, 0, cv::INTER_NEAREST);
    }

    src3.convertTo(src3, CV_8UC3);
    // Вызываем watershed.
    try {
        cv::watershed(src3, markers);
    } catch (const cv::Exception& e) {
        std::cerr << "cv::watershed exception: " << e.what() << std::endl;
        throw;
    }

    // Собираем уникальные метки для определения нового значения для границ.
    std::set<int> uniqueMarkers;
    for (int y = 0; y < markers.rows; y++) {
        for (int x = 0; x < markers.cols; x++) {
            uniqueMarkers.insert(markers.at<int>(y, x));
        }
    }
    int newLabel = static_cast<int>(uniqueMarkers.size()) - 1;
    for (int y = 0; y < markers.rows; y++) {
        for (int x = 0; x < markers.cols; x++) {
            int val = markers.at<int>(y, x);
            if (val == -1)
                markers.at<int>(y, x) = newLabel;
            if (val == 255)
                markers.at<int>(y, x) = 0;
        }
    }

    if (!markers.empty()) {
        cv::Mat temp;
        markers.convertTo(temp, CV_32F);
        cv::imshow("watershed labels", temp);
    }
    cv::Mat segments;
    markers.convertTo(segments, CV_8U);
    return segments;
}

