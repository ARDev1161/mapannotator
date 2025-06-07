#ifndef UTILS_H
#define UTILS_H

#include <stdexcept>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>

static struct MapInfo{
    double resolution;
    double originX;
    double originY;
    double theta;
    int height;
    int width;
} mapInfo;

// Функция для преобразования координат из пикселя в реальные координаты.
// Параметры:
// - pixel: координата в пикселях (изображение имеет начало координат в верхнем левом угле)
// - resolution: размер одного пикселя в метрах
// - origin: вектор из трёх значений [origin_x, origin_y, origin_theta],
//           где (origin_x, origin_y) – мировые координаты нижнего левого угла карты,
//           а origin_theta – угол поворота (в радианах)
// - image_height: высота изображения (в пикселях), чтобы правильно инвертировать ось Y
static cv::Point2d pixelToWorld(const cv::Point2d & pixel, const MapInfo & mapParams) {
    // Перевод пиксельных координат в координаты карты (относительно нижнего левого угла)
    // Добавляем 0.5 для получения центра пикселя
    double x_map = (pixel.x + 0.5) * mapParams.resolution;
    double y_map = ((mapParams.height - pixel.y) - 0.5) * mapParams.resolution;

    // Применяем поворот: учитываем, что карта может быть повернута относительно мировых координат
    double world_x = mapParams.originX + x_map * cos(mapParams.theta) - y_map * sin(mapParams.theta);
    double world_y = mapParams.originY + x_map * sin(mapParams.theta) + y_map * cos(mapParams.theta);
    return cv::Point2d(world_x, world_y);
}

// Загрузка карты в формате uint8 (BGR).
static cv::Mat loadMapFromFileUint8(const std::filesystem::path& path) {
    cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Could not load image: " + path.string());
    }
    return img;
}

// Загрузка карты в формате float32 с масштабированием [0,1].
static cv::Mat loadMapFromFile(const std::filesystem::path& path) {
    cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Could not load image: " + path.string());
    }
    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_32FC3, 1.0/255.0);
    return imgFloat;
}

// Загрузка карты из буфера.
static cv::Mat loadMapFromBuffer(const std::vector<uchar>& buffer) {
    cv::Mat img = cv::imdecode(buffer, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Could not decode image from buffer.");
    }
    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_32FC3, 1.0/255.0);
    return imgFloat;
}

// Преобразование 3-канального изображения в градационное (grayscale).
static cv::Mat makeGray(const cv::Mat& src) {
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

// Загрузка градационного изображения из файла.
static cv::Mat loadGrayMapFromFile(const std::filesystem::path& path) {
    cv::Mat img = loadMapFromFile(path);
    return makeGray(img);
}

// Загрузка градационного изображения из буфера.
static cv::Mat loadGrayMapFromBuffer(const std::vector<uchar>& buffer) {
    cv::Mat img = loadMapFromBuffer(buffer);
    return makeGray(img);
}

// Инвертирование изображения.
static cv::Mat makeInvert(const cv::Mat& src) {
    cv::Mat result;
    if (src.type() == CV_8U) {
        result = 255 - src;
        return result;
    } else if (src.type() == CV_32F) {
        result = 1.0 - src;
        return result;
    } else {
        throw std::runtime_error("makeInvert does not support the provided type.");
    }
}

// Бинаризация изображения с порогом.
static cv::Mat makeBinary(const cv::Mat& src, double threshold, double maxValue = 1.0) {
    cv::Mat binary;
    cv::threshold(src, binary, threshold, maxValue, cv::THRESH_BINARY);
    return binary;
}

// Дилатация бинарного изображения.
static cv::Mat dilateBinary(const cv::Mat& src, int kernel_size = 3, int iterations = 1) {
    CV_Assert(src.depth() == CV_8U ||
              src.depth() == CV_16U ||
              src.depth() == CV_16S ||
              src.depth() == CV_32F ||
              src.depth() == CV_64F);
    cv::Mat result;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::dilate(src, result, kernel, cv::Point(-1, -1), iterations);
    return result;
}

// Эрозия бинарного изображения.
static cv::Mat erodeBinary(const cv::Mat& src, int kernel_size = 3, int iterations = 1) {
    CV_Assert(src.depth() == CV_8U ||
              src.depth() == CV_16U ||
              src.depth() == CV_16S ||
              src.depth() == CV_32F ||
              src.depth() == CV_64F);
    cv::Mat result;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::erode(src, result, kernel, cv::Point(-1, -1), iterations);
    return result;
}

// Функция для вычисления kernel size по sigma: обычно используется правило 2*ceil(3*sigma)+1
static int getKernelSize(double sigma) {
    return static_cast<int>(2 * std::ceil(3 * sigma) + 1);
}

static std::string type2str(int type) {
    std::string r;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }
    r += "C";
    r += std::to_string(chans);
    return r;
}

/**
 *  Конвертирует произвольный cv::Mat -> CV_8U, 1 или 3 канала.
 *
 *  @param  src             любая матрица (глубина 8/16/32, 1–4 канала)
 *  @param  applyColorMap   true  => к 1‑канальному применяем COLORMAP_JET
 *  @param  dst             результат, которым можно сразу imshow(...)
 */
static inline bool toDisplayable(const cv::Mat& src,
                          cv::Mat&       dst,
                          bool           applyColorMap = false)
{
    if(src.empty())
        return false;

    cv::Mat tmp;

    /* ---------- приведём глубину к 8‑бит -------------------------------- */
    int depth = src.depth();
    if (depth == CV_8U) {
        tmp = src;                    // уже 8‑бит
    } else {
        double minVal, maxVal;
        cv::minMaxLoc(src, &minVal, &maxVal);
        if (maxVal - minVal < 1e-12) maxVal = minVal + 1.0; // защита

        double scale = 255.0 / (maxVal - minVal);
        double shift = -minVal * scale;
        src.convertTo(tmp, CV_8U, scale, shift);
    }

    /* ---------- упростим число каналов ---------------------------------- */
    int ch = tmp.channels();

    if (ch == 1) {
        if (applyColorMap) {
            cv::applyColorMap(tmp, dst, cv::COLORMAP_JET); // CV_8UC3
        } else {
            dst = tmp;                                     // CV_8UC1
        }
    } else if (ch == 3) {
        dst = tmp;                                         // CV_8UC3
    } else if (ch == 4) {                                  // BGRA -> BGR
        cv::cvtColor(tmp, dst, cv::COLOR_BGRA2BGR);
    } else {
        /* например, CV_32FC2: сведём к среднему по каналам */
        cv::Mat gray;
        std::vector<cv::Mat> planes;
        cv::split(tmp, planes);
        cv::Mat acc = cv::Mat::zeros(tmp.size(), CV_32F);
        for (auto& p : planes) {
            p.convertTo(p, CV_32F);
            acc += p;
        }
        acc /= static_cast<float>(planes.size());
        acc.convertTo(gray, CV_8U);
        if (applyColorMap)
            cv::applyColorMap(gray, dst, cv::COLORMAP_JET);
        else
            dst = gray;
    }

    return true;
}

static void showMat(const std::string &windowName, const cv::Mat &mat, bool isColor = true)
{
    cv::Mat vis;
    if(toDisplayable(mat, vis, isColor))
    {
        cv::namedWindow(windowName, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::resizeWindow(windowName, mat.cols, mat.rows);
        cv::imshow(windowName, mat);
    }

}

/*-------------------------------------------------------------------------*/
/*  service helpers                                                        */
/*-------------------------------------------------------------------------*/

/// Читаемый вывод cv::Mat::type()
static std::string matTypeStr(int t)
{
    const int depth = t & CV_MAT_DEPTH_MASK;
    const int chans = 1 + (t >> CV_CN_SHIFT);

    const char* depthStr =
        depth == CV_8U  ? "CV_8U"  :
        depth == CV_8S  ? "CV_8S"  :
        depth == CV_16U ? "CV_16U" :
        depth == CV_16S ? "CV_16S" :
        depth == CV_32S ? "CV_32S" :
        depth == CV_32F ? "CV_32F" :
        depth == CV_64F ? "CV_64F" : "UNKNOWN";

    std::ostringstream oss;
    oss << depthStr << 'C' << chans;
    return oss.str();
}

/**
 * @brief Computes the real-world area represented by white pixels in a binary mask.
 *
 * @param mask           Binary image (type CV_8UC1). Pixels that contribute to the area
 *                       must be exactly 255; everything else (0) is ignored.
 * @param m2_per_pixel   Conversion factor: physical area (m²) represented by each pixel.
 *                       For example, if the map resolution is 0.05 m × 0.05 m, pass
 *                       0.05 * 0.05 = 0.0025.
 * @return               Area in square metres.
 */
inline double computeWhiteArea(const cv::Mat1b * mask, double m2_per_pixel)
{
    if(!mask)
        return 0.0;

    CV_Assert(!mask->empty() && mask->type() == CV_8UC1);
    CV_Assert(m2_per_pixel > 0.0);

    // Fast, vectorised popcount of non-zero elements (each non-zero byte is treated as 1)
    const int whitePixelCount = cv::countNonZero(*mask);

    return whitePixelCount * m2_per_pixel;
}

/*-------------------------------------------------------------------------*/
/*  проверка совместимости                                                 */
/*-------------------------------------------------------------------------*/
/**
 * @brief   Проверить, одинаковы ли размеры и типы набора матриц.
 * @param   mats  список матриц (2 и более)
 * @return  0  – совместимы
 *         -1  – различаются размеры
 *         -2  – различаются типы
 *
 * В случае несовместимости вся информация выводится в std::cerr.
 */
static int checkMatCompatibility(const std::vector<cv::Mat>& mats)
{
    if (mats.size() < 2)               // одна — уж точно «совместима»
        return 0;

    const cv::Size refSize = mats[0].size();
    const int      refType = mats[0].type();

    bool sizeMismatch = false;
    bool typeMismatch = false;

    for (size_t i = 1; i < mats.size(); ++i)
    {
        if (mats[i].size() != refSize) sizeMismatch = true;
        if (mats[i].type() != refType) typeMismatch = true;
    }

    if (!sizeMismatch && !typeMismatch)
        return 0;                      // всё ок

    /* —―― выводим подробности ――― */
    std::cerr << "[checkMatCompatibility] mismatch detected:\n";
    for (size_t i = 0; i < mats.size(); ++i)
        std::cerr << "  #" << i << ": size = " << mats[i].cols << 'x' << mats[i].rows
                  << ", type = " << matTypeStr(mats[i].type()) << '\n';

    return sizeMismatch ? -1 : -2;
}

/*-------------------------------------------------------------------------*/
/*  удобный вариадик-обёртка — можно передавать любое число матриц         */
/*-------------------------------------------------------------------------*/
template<typename... Mats>
int checkMatCompatibility(const cv::Mat& m0, const cv::Mat& m1, const Mats&... rest)
{
    std::vector<cv::Mat> pack = { m0, m1, rest... };
    return checkMatCompatibility(pack);
}

#endif // UTILS_H
