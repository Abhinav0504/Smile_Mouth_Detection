#ifndef RENDER_H
#define RENDER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "Types.h"

// Define Color struct
struct Color {
    int r, g, b, a = -1; // Default alpha as -1 (indicating no alpha)

    cv::Scalar asScalar() const;
};

// Predefined Colors
struct Colors {
    static const Color BLACK;
    static const Color RED;
    static const Color GREEN;
    static const Color BLUE;
    static const Color PINK;
    static const Color WHITE;
};

// Define Point struct
struct Point {
    float x, y;

    cv::Point2f asCVPoint() const;
};

// Define RectOrOval struct for bounding boxes
struct RectOrOval {
    float left, top, right, bottom;
    bool oval;

    cv::Rect asCVRect() const;
};


// Define Annotation struct for rendering
struct Annotation {
    std::vector<Point> points;
    RectOrOval rect;
    Color color;
    bool normalized_positions;
    int thickness;
};

// Function declarations
std::vector<Annotation> detections_to_render_data(
        const std::vector<Detection>& detections,
        const Color& bounds_color = Colors::RED,
        const Color& keypoint_color = Colors::BLUE,
        int line_width = 2,
        int point_width = 4,
        bool normalized_positions = true);

cv::Mat render_to_image(const std::vector<Annotation>& annotations, cv::Mat& image);

#endif // RENDER_H
