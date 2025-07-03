#include "mappreprocessing.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <queue>

// Функция находит корректировку ориентации карты по контурам.
// Если изображение имеет тип CV_32F (значения [0,1]), оно преобразуется в CV_8U.
cv::RotatedRect MapPreprocessing::findMapOrientationAdjustment(const cv::Mat& im, double& angle) {
    cv::Mat im8;
    if (im.type() == CV_32F) {
        im.convertTo(im8, CV_8U, 255.0);
    } else {
        im8 = im;
    }
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(im8, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // Если контуров меньше двух, возвращаем пустой прямоугольник и угол 0.
    if (contours.size() < 2) {
        angle = 0.0;
        return cv::RotatedRect();
    }
    // Сортируем контуры по площади.
    std::sort(contours.begin(), contours.end(),
              [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                  return cv::contourArea(a) < cv::contourArea(b);
              });
    // Выбираем предпоследний контур (второй по величине).
    cv::RotatedRect rect = cv::minAreaRect(contours[contours.size() - 2]);
    angle = rect.angle;
    return rect;
}

// Поворачивает изображение на заданный угол.
// Фоновое значение берётся из первого пикселя исходного изображения.
cv::Mat MapPreprocessing::rotateImage(const cv::Mat& image, double angle) {
    cv::Point2f center(image.cols / 2.0F, image.rows / 2.0F);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat result;

    // Определяем значение фона на основе первого пикселя.
    cv::Scalar borderValue;
    if (image.channels() == 1) {
        if (image.type() == CV_8U)
            borderValue = cv::Scalar(image.at<uchar>(0, 0));
        else if (image.type() == CV_32F)
            borderValue = cv::Scalar(image.at<float>(0, 0));
    } else {
        if (image.type() == CV_8UC3)
            borderValue = cv::Scalar(image.at<cv::Vec3b>(0, 0));
        else if (image.type() == CV_32FC3)
            borderValue = cv::Scalar(image.at<cv::Vec3f>(0, 0));
    }
    cv::warpAffine(image, result, rot_mat, image.size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, borderValue);

    // Если входное изображение было CV_32F, возвращаем результат в том же формате.
    if (image.type() == CV_32F) {
        cv::Mat resultFloat;
        result.convertTo(resultFloat, CV_32F, 1.0 / 255.0);
        return resultFloat;
    }
    return result;
}

// Отрисовывает ограничивающий прямоугольник (bounding box) на изображении.
cv::Mat MapPreprocessing::drawBB(const cv::Mat& im, const cv::RotatedRect& rect, const cv::Scalar& color) {
    cv::Mat output;
    if (im.channels() == 1)
        cv::cvtColor(im, output, cv::COLOR_GRAY2BGR);
    else
        output = im.clone();

    cv::Point2f vertices[4];
    rect.points(vertices);
    std::vector<cv::Point> pts;
    for (int i = 0; i < 4; ++i)
        pts.push_back(vertices[i]);

    const cv::Point* pts_array = pts.data();
    int npts = 4;
    cv::polylines(output, &pts_array, &npts, 1, true, color, 2);
    return output;
}

// Рисует линию на изображении.
void MapPreprocessing::drawLine(cv::Mat& im, int x0, int y0, int x1, int y1,
                            const cv::Scalar& color, int lineThickness) {
    cv::line(im, cv::Point(x0, y0), cv::Point(x1, y1), color, lineThickness);
}

// Вычисляет взвешенную гистограмму углов по изображению rank.
// Используются cv::Canny и cv::HoughLinesP (вероятностное преобразование Хафа).
std::vector<double> MapPreprocessing::computeWeightedAngleHistogram(const cv::Mat& rank,
                                                                const AlignmentConfig& alignmentConfig,
                                                                bool debug) {
    cv::Mat edges;
    cv::Mat rank8;
    if (rank.type() == CV_32F)
        rank.convertTo(rank8, CV_8U, 255.0);
    else
        rank8 = rank;

    cv::Canny(rank8, edges,
              alignmentConfig.cannyConfig.lowThreshold,
              alignmentConfig.cannyConfig.highThreshold,
              alignmentConfig.cannyConfig.apertureSize,
              alignmentConfig.cannyConfig.L2gradient);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines,
                    alignmentConfig.houghConfig.rho,
                    alignmentConfig.houghConfig.theta,
                    alignmentConfig.houghConfig.threshold,
                    alignmentConfig.houghConfig.minLineLength,
                    alignmentConfig.houghConfig.maxLineGap);

    cv::Mat linesImg;
    if (debug)
        linesImg = cv::Mat::zeros(rank.size(), CV_8UC3);
    auto res = alignmentConfig.histogramConfig.resolution;
    std::vector<double> hist(res, 0.0);
    for (const auto& line : lines) {
        int x0 = line[0], y0 = line[1], x1 = line[2], y1 = line[3];
        double angle = std::atan2(static_cast<double>(y1 - y0),
                                  static_cast<double>(x1 - x0)) * 180.0 / CV_PI;
        double angle_90 = std::fmod(angle, 90.0);
        if (angle_90 < 0) angle_90 += 90.0;
        if (debug) {
            cv::Scalar lineColor(255 * angle_90 / 90.0, 0, 0);
            drawLine(linesImg, x0, y0, x1, y1, lineColor, 1);
        }
        int angle_bin = static_cast<int>(std::floor(angle_90 / 90.0 * res));
        if (angle_bin < 0)
            angle_bin = 0;
        if (angle_bin >= res)
            angle_bin = res - 1;
        double length = std::sqrt(std::pow(x1 - x0, 2) + std::pow(y1 - y0, 2));
        hist[angle_bin] += length;
    }
    return hist;
}

// По гистограмме находит лучший угол выравнивания с использованием скользящего среднего.
double MapPreprocessing::findBestAlignmentAngleFromHistogram(const std::vector<double>& hist,
                                                         const HistogramConfig& histConfig) {
    int k = histConfig.windowHalfSize;
    int N = 2 * k + 1;
    int size = static_cast<int>(hist.size());

    // Формируем расширенную гистограмму: последние k элементов, затем исходная гистограмма, затем первые k.
    std::vector<double> augmented;
    augmented.insert(augmented.end(), hist.end() - k, hist.end());
    augmented.insert(augmented.end(), hist.begin(), hist.end());
    augmented.insert(augmented.end(), hist.begin(), hist.begin() + k);

    // Вычисляем скользящее среднее.
    std::vector<double> slidingAverage;
    for (size_t i = 0; i <= augmented.size() - N; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < static_cast<size_t>(N); j++) {
            sum += augmented[i + j];
        }
        slidingAverage.push_back(sum / N);
    }

    // Находим индекс максимального среднего.
    int maxIndex = 0;
    double maxVal = slidingAverage[0];
    for (size_t i = 1; i < slidingAverage.size(); i++) {
        if (slidingAverage[i] > maxVal) {
            maxVal = slidingAverage[i];
            maxIndex = static_cast<int>(i);
        }
    }

    // Вычисляем угол: индекс делим на разрешение и умножаем на 90.
    double theta_deg = (static_cast<double>(maxIndex) / histConfig.resolution) * 90.0;
    if (theta_deg > 45.0)
        theta_deg -= 90.0;
    return theta_deg;
}

// Полный пайплайн выравнивания.
// В данном примере предварительная обработка реализована простым пороговым преобразованием.
double MapPreprocessing::findAlignmentAngle(const cv::Mat& grayscale, const AlignmentConfig& config) {
    cv::Mat binary;
    if (grayscale.type() != CV_8U)
        grayscale.convertTo(binary, CV_8U, 255.0);
    else
        binary = grayscale.clone();
    cv::threshold(binary, binary, 128, 255, cv::THRESH_BINARY);

    std::vector<double> hist = computeWeightedAngleHistogram(binary, config, false);
    double angle_deg = findBestAlignmentAngleFromHistogram(hist, config.histogramConfig);
    return angle_deg;
}

cv::Mat MapPreprocessing::unknownRegionsDissolution(const cv::Mat& src,
                                                     int kernelSize,
                                                     int maxIter)
{
    CV_Assert(!src.empty());

    int type = src.type();
    cv::Mat current;
    if (type != CV_8U)
        src.convertTo(current, CV_8U, 255.0);
    else
        current = src.clone();

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                               cv::Size(kernelSize, kernelSize));

    int prevGray = -1;
    for (int i = 0; i < maxIter; ++i)
    {
        cv::erode(current, current, kernel);
        cv::dilate(current, current, kernel);

        cv::Mat grayMask;
        cv::inRange(current, 1, 254, grayMask); // pixels that are neither black nor white
        int grayCount = cv::countNonZero(grayMask);

        if (grayCount == prevGray)
            break;
        prevGray = grayCount;
    }

    if (type != CV_8U)
    {
        cv::Mat res;
        current.convertTo(res, type, 1.0 / 255.0);
        return res;
    }

    return current;
}

cv::Mat MapPreprocessing::removeGrayIslands(const cv::Mat& src, int connectivity)
{
    CV_Assert(!src.empty());
    CV_Assert(connectivity == 4 || connectivity == 8);

    int type = src.type();
    cv::Mat work;
    if (type != CV_8U)
        src.convertTo(work, CV_8U, 255.0);
    else
        work = src.clone();

    cv::Mat visited = cv::Mat::zeros(work.size(), CV_8U);
    std::queue<cv::Point> q;
    for (int y = 0; y < work.rows; ++y)
    {
        for (int x = 0; x < work.cols; ++x)
        {
            if (work.at<uchar>(y, x) == 0)
            {
                visited.at<uchar>(y, x) = 1;
                q.emplace(x, y);
            }
        }
    }

    std::vector<cv::Point> dirs;
    if (connectivity == 4)
        dirs = { {1,0}, {-1,0}, {0,1}, {0,-1} };
    else
        dirs = { {1,0}, {-1,0}, {0,1}, {0,-1}, {1,1}, {1,-1}, {-1,1}, {-1,-1} };

    while (!q.empty())
    {
        cv::Point p = q.front();
        q.pop();
        for (const auto& d : dirs)
        {
            int nx = p.x + d.x;
            int ny = p.y + d.y;
            if (nx < 0 || ny < 0 || nx >= work.cols || ny >= work.rows)
                continue;
            if (visited.at<uchar>(ny, nx))
                continue;
            uchar val = work.at<uchar>(ny, nx);
            if (val == 255)
                continue;
            visited.at<uchar>(ny, nx) = 1;
            q.emplace(nx, ny);
        }
    }

    cv::Mat result = work.clone();
    for (int y = 0; y < work.rows; ++y)
    {
        for (int x = 0; x < work.cols; ++x)
        {
            uchar val = work.at<uchar>(y, x);
            if (val != 0 && val != 255 && !visited.at<uchar>(y, x))
                result.at<uchar>(y, x) = 255;
        }
    }

    if (type != CV_8U)
    {
        cv::Mat res;
        result.convertTo(res, type, 1.0 / 255.0);
        return res;
    }

    return result;
}

// Реализация генерации денойзенной карты.
// Здесь вызываются функции из Segmentation для бинаризации, кадрирования, инверсии и удаления шумовых компонентов.
std::pair<cv::Mat, Segmentation::CropInfo> MapPreprocessing::generateDenoisedAlone(const cv::Mat& raw,
    const DenoiseConfig& config) {

    // Устраняем серые (неизвестные) зоны перед дальнейшей обработкой
    cv::Mat preprocessed = unknownRegionsDissolution(raw);

    // Создаем бинарную карту для определения области кадрирования.
    cv::Mat binaryForCrop = makeBinary(preprocessed, config.binaryForCropThreshold * 255, 255);
    Segmentation::CropInfo cropInfo = Segmentation::cropSingleInfo(binaryForCrop, config.cropPadding);

    // Освобождаем память, если необходимо.
    // Обрезаем исходное изображение до ROI.
    cv::Mat rank = Segmentation::cropBackground(preprocessed, cropInfo);
    rank = removeGrayIslands(rank);

    // Применяем бинаризацию для удаления шума.
    rank = makeBinary(rank, config.binaryThreshold * 255, 255);

    // Инвертируем изображение.
    rank = makeInvert(rank);
    // Удаляем маленькие компоненты (снаружи).
    rank = Segmentation::removeSmallConnectedComponents(rank, "fixed", config.compOutMinSize, 4, false);
    // Инвертируем снова.
    rank = makeInvert(rank);
    // Удаляем маленькие компоненты (внутри).
    rank = Segmentation::removeSmallConnectedComponents(rank, "fixed", config.compInMinSize, 4, false);

    // Финальная бинаризация.
    rank = makeBinary(rank, config.rankBinaryThreshold);
    return { rank, cropInfo };
}

bool MapPreprocessing::mapAlign(const cv::Mat& raw, cv::Mat& out, const AlignmentConfig& config){
    double alignmentAngle = 0.0;
    if (config.enable) {
        // Находим угол выравнивания с помощью MapAlignment.
        alignmentAngle = MapPreprocessing::findAlignmentAngle(raw, config);
        out = MapPreprocessing::rotateImage(raw, alignmentAngle);
        return true;
    }

    return false;
}
