#pragma once
#include <string>
#include <vector>

    /** Configuration for the Canny edge detector. */
    struct CannyConfig {
        double lowThreshold = 100; ///< lower hysteresis threshold
        double highThreshold = 150; ///< upper hysteresis threshold
        int apertureSize = 3; ///< Sobel aperture size
        bool L2gradient = false; ///< use more accurate L2 norm
    };

    /** Configuration for the probabilistic Hough transform. */
    struct HoughConfig {
        double rho = 1;                  ///< distance resolution of the accumulator
        double theta = 0.01745;          ///< angle resolution in radians (e.g. CV_PI/180)
        int threshold = 50;              ///< accumulator threshold
        double minLineLength = 50;       ///< minimum line length
        double maxLineGap = 10;          ///< maximum allowed gap between points
    };

    /** Parameters for computing the angle histogram. */
    struct HistogramConfig {
        int resolution = 90;      ///< number of histogram bins
        int windowHalfSize = 2;   ///< half window size for moving average
    };

    /**
     * Configuration for the alignment pipeline. Preprocessing (binarisation,
     * cropping, etc.) is expected to be done outside of this class.
     */
    struct AlignmentConfig {
        bool enable = true;          ///< enable alignment stage
        CannyConfig cannyConfig;     ///< parameters for Canny
        HoughConfig houghConfig;     ///< parameters for Hough transform
        HistogramConfig histogramConfig; ///< histogram parameters
    };

    /** Configuration for the denoising stage (generate_denoised_alone). */
    struct DenoiseConfig {
        double binaryForCropThreshold = 0.2; ///< threshold for cropping mask
        int cropPadding;                     ///< padding when cropping
        double binaryThreshold = 0.2;        ///< threshold for noise removal
        int compOutMinSize = 40;             ///< minimal size of outer components
        int compInMinSize = 40;              ///< minimal size of inner components
        double rankBinaryThreshold;          ///< final binarisation threshold
    };

    /** Configuration for dilation. */
    struct DilateConfig {
        int kernelSize = 3; ///< kernel size for dilation
        int iterations = 1; ///< number of iterations
    };

    /** Configuration for ridge map generation. */
    struct RidgeConfig {
        std::string mode = "";           ///< generation mode
        std::vector<double> sigmas = {1.0, 2.0, 3.0}; ///< list of scales
    };

    /** Parameters used for computing the label list. */
    struct LabelsListConfig {
        double sigmaStart = 1.0;              ///< starting sigma
        double sigmaStep = 0.2;               ///< step between sigma values
        double gaussianSeedsThreshold = 0.05; ///< threshold for seeds
        int maxIter = 1;                      ///< number of iterations
        int backgroundErosionKernelSize = 3;  ///< erosion kernel size
        bool debug = true;                    ///< enable debug output
    };

    /** Configuration for the over-segmentation stage. */
    struct OverSegmentConfig {
        // Additional parameters for map_initial_seeds or final watershed can be
        // added here.
    };

    /** Configuration for node merging stage. */
    struct MergeNodesConfig {
        double areaThreshold = 10.0; ///< merge zones smaller than this area
    };

    /** Configuration for edge merging stage. */
    struct MergeEdgesConfig {
        double lengthThreshold; ///< merge edges shorter than this length
    };

    /** Combined configuration for the full segmentation pipeline. */
    struct SegmenterConfig {
        AlignmentConfig alignmentConfig;   ///< alignment parameters
        DenoiseConfig denoiseConfig;       ///< denoising parameters
        DilateConfig dilateConfig;         ///< dilation parameters
        RidgeConfig ridgeConfig;           ///< ridge map parameters
        LabelsListConfig labelsListConfig; ///< label list computation
        OverSegmentConfig overSegmentConfig; ///< over-segmentation parameters
        MergeNodesConfig mergeNodesConfig;   ///< node merge parameters
        MergeEdgesConfig mergeEdgesConfig;   ///< edge merge parameters
    };
