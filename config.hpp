#pragma once
#include <string>
#include <vector>

    // Конфигурация для Canny.
    struct CannyConfig {
        double lowThreshold = 100;
        double highThreshold = 150;
        int apertureSize = 3;
        bool L2gradient = false;
    };

    // Конфигурация для вероятностного преобразования Хафа.
    struct HoughConfig {
        double rho = 1;
        double theta = 0.01745; // в радианах (например, CV_PI/180)
        int threshold = 50;
        double minLineLength = 50;
        double maxLineGap = 10;
    };

    // Конфигурация для вычисления гистограммы углов.
    struct HistogramConfig {
        int resolution = 90;      // количество бинов гистограммы
        int windowHalfSize = 2;  // половина окна для скользящего среднего
    };

    // Конфигурация для полного пайплайна выравнивания.
    // Предполагается, что предварительная обработка (бинаризация, crop и т.д.) производится вне класса.
    struct AlignmentConfig {
        bool enable = true;  // выполнять ли выравнивание
        CannyConfig cannyConfig;
        HoughConfig houghConfig;
        HistogramConfig histogramConfig;
    };

    // Конфигурация для этапа денойзинга (generate_denoised_alone).
    struct DenoiseConfig {
        double binaryForCropThreshold = 0.2;
        int cropPadding;
        double binaryThreshold = 0.2;
        // Параметры для удаления маленьких компонентов (components_out и components_in).
        int compOutMinSize = 40;
        int compInMinSize = 40;
        // Порог для финальной бинаризации (rank/make_binary).
        double rankBinaryThreshold;
    };

    // Конфигурация для дилатации.
    struct DilateConfig {
        int kernelSize = 3;
        int iterations = 1;
    };

    // Конфигурация для генерации карты гребней.
    struct RidgeConfig {
        std::string mode = "";
        std::vector<double> sigmas = {1.0, 2.0, 3.0};
    };

    // Конфигурация для вычисления списка меток.
    struct LabelsListConfig {
        double sigmaStart = 1.0;
        double sigmaStep = 0.2;
        double gaussianSeedsThreshold = 0.05;
        int maxIter = 1;
        int backgroundErosionKernelSize = 3;
        bool debug = true;
    };

    // Конфигурация для этапа over-segmentation.
    struct OverSegmentConfig {
        // Здесь можно добавить параметры для генерации map_initial_seeds,
        // а также для создания финальной watershed-сегментации.
    };

    // Конфигурация для этапа слияния узлов.
    struct MergeNodesConfig {
        double areaThreshold = 10.0;
    };

    // Конфигурация для этапа слияния рёбер.
    struct MergeEdgesConfig {
        double lengthThreshold;
    };

    // Объединённая конфигурация пайплайна сегментации.
    struct SegmenterConfig {
        AlignmentConfig alignmentConfig;
        DenoiseConfig denoiseConfig;
        DilateConfig dilateConfig;
        RidgeConfig ridgeConfig;
        LabelsListConfig labelsListConfig;
        OverSegmentConfig overSegmentConfig;
        MergeNodesConfig mergeNodesConfig;
        MergeEdgesConfig mergeEdgesConfig;
    };
