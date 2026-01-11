#ifndef ENDPOINTS_H
#define ENDPOINTS_H

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

static cv::Mat1b getSkeletonMat(const cv::Mat1b &wallMask)
{
    cv::Mat1b skeleton;
    cv::ximgproc::thinning(wallMask, skeleton, cv::ximgproc::THINNING_ZHANGSUEN);
    return skeleton;
}

/**
 * @brief Возвращает цветную картинку: белый скелет + зелёные концевые точки.
 *        Одновременно собирает сами точки (endpoints) во внешний вектор.
 *
 * @param skeleton      CV_8UC1 карта-скелет (0 / 255).
 * @param endpointsOut  (необязательно) куда записать найденные точки.
 * @return cv::Mat3b    BGR-изображение для отладки.
 */
static cv::Mat3b visualizeSkeletonEndpoints(const cv::Mat1b& skeleton,
                                     std::vector<cv::Point>* endpointsOut = nullptr)
{
    CV_Assert(!skeleton.empty() && skeleton.type() == CV_8UC1);

    const int rows = skeleton.rows, cols = skeleton.cols;

    /* — 1. По скелету ищем пиксели степени 1 ———————————————— */
    std::vector<cv::Point> endpoints;
    endpoints.reserve(256);                             // разумное начальное число

    auto inside = [&](int y, int x)
    {
        return (0 <= y && y < rows && 0 <= x && x < cols);
    };

    for (int y = 0; y < rows; ++y)
    {
        const uchar* rowPtr = skeleton.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x, ++rowPtr)
        {
            if (*rowPtr == 0) continue;                 // фоновый пиксель

            int neigh = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                {
                    if (dx == 0 && dy == 0) continue;
                    int yy = y + dy, xx = x + dx;
                    if (inside(yy, xx) && skeleton(yy, xx))
                        ++neigh;
                }

            if (neigh == 1)                            // степень-1  ⇒ конец линии
                endpoints.emplace_back(x, y);
        }
    }

    if (endpointsOut) *endpointsOut = endpoints;       // вернуть список, если нужно

    /* — 2. Визуализация: переводим скелет в BGR и рисуем точки ———————— */
    cv::Mat3b vis;
    cv::cvtColor(skeleton, vis, cv::COLOR_GRAY2BGR);    // 0/255 → чёрно-белый BGR

    for (const auto& p : endpoints)
        cv::circle(vis, p, 2, cv::Scalar(0, 255, 0), cv::FILLED);   // зелёный Ø≈5 px

    return vis;
}

/**
 * @brief Возвращает все концевые точки скелета (пиксели степени 1).
 *
 * @param skeleton  CV_8UC1 карта-скелет: 0 — фон, ненулевое — линия.
 * @return std::vector<cv::Point>  Список координат в формате (x, y).
 */
static std::vector<cv::Point> findSkeletonEndpoints(const cv::Mat1b& skeleton)
{
    CV_Assert(!skeleton.empty() && skeleton.type() == CV_8UC1);

    const int rows = skeleton.rows, cols = skeleton.cols;
    std::vector<cv::Point> endpoints;
    endpoints.reserve(256);                       // начальный запас

    auto inside = [&](int y, int x)
    {
        return (0 <= y && y < rows && 0 <= x && x < cols);
    };

    for (int y = 0; y < rows; ++y)
    {
        const uchar* rowPtr = skeleton.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x, ++rowPtr)
        {
            if (*rowPtr == 0)                     // фон
                continue;

            int neigh = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                {
                    if (dx == 0 && dy == 0) continue;
                    int yy = y + dy, xx = x + dx;
                    if (inside(yy, xx) && skeleton(yy, xx))
                        ++neigh;
                }

            if (neigh == 1)                       // степень-1 ⇒ конец линии
                endpoints.emplace_back(x, y);
        }
    }
    return endpoints;
}

/**
 * @brief Строит карту-визуализацию концов скелета.
 *        Белый фон (255), концевые точки — чёрные круги указанного радиуса.
 *
 * @param skeleton  CV_8UC1: 0 — фон, ненулевое — линия скелета
 * @param radius    Радиус кружка (>= 1) для визуализации точки
 * @return cv::Mat1b  Одноканальная матрица тех же размеров, что и skeleton
 */
static cv::Mat1b drawSkeletonEndpoints(const cv::Mat1b& skeleton, int radius)
{
    CV_Assert(!skeleton.empty() && skeleton.type() == CV_8UC1);
    CV_Assert(radius >= 1);

    const int rows = skeleton.rows, cols = skeleton.cols;

    /* — подготовим «чистый лист» ——————————————— */
    cv::Mat1b canvas(rows, cols, uchar(255));      // фон = белый (255)

    auto inside = [&](int y, int x)
    {
        return (0 <= y && y < rows && 0 <= x && x < cols);
    };

    /* — ищем и сразу рисуем концевые пиксели ———— */
    for (int y = 0; y < rows; ++y)
    {
        const uchar* rowPtr = skeleton.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x, ++rowPtr)
        {
            if (*rowPtr == 0) continue;            // не линия

            int neigh = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                {
                    if (dx == 0 && dy == 0) continue;
                    int yy = y + dy, xx = x + dx;
                    if (inside(yy, xx) && skeleton(yy, xx))
                        ++neigh;
                }

            if (neigh == 1)                       // степень-1 ⇒ конец линии
                cv::circle(canvas,                 // рисуем чёрным
                           {x, y},
                           radius,
                           cv::Scalar(0),          // цвет 0 (чёрный)
                           cv::FILLED);
        }
    }

    return canvas;                                // белый фон + чёрные точки
}

#endif // ENDPOINTS_H
