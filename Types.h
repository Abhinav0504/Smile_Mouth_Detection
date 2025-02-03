//#ifndef TYPES_H
//#define TYPES_H
//
//#include <opencv2/opencv.hpp>
//#include <vector>
//#include <array>
//#include <cmath>
//#include <optional>
//
//// ImageTensor class declaration
//class ImageTensor {
//public:
//    cv::Mat tensor_data;
//    std::array<float, 4> padding;
//    cv::Size original_size;
//
//    ImageTensor(const cv::Mat& data, const std::array<float, 4>& pad, const cv::Size& original);
//};
//
//// Rect class declaration
//class Rect {
//public:
//    float x_center, y_center, width, height, rotation;
//    bool normalized;
//
//    Rect(float x, float y, float w, float h, float r, bool norm);
//    std::array<float, 2> size() const;
//    Rect scaled(const std::array<float, 2>& size, bool normalize = false) const;
//    std::vector<std::array<float, 2>> points() const;
//};
//
//// BBox class declaration
//class BBox {
//public:
//    float xmin, ymin, xmax, ymax;
//
//    BBox(float x1, float y1, float x2, float y2);
//    std::array<float, 4> as_tuple() const;
//    float width() const;
//    float height() const;
//    bool is_empty() const;
//    bool normalized() const;
//    float area() const;
//    std::optional<BBox> intersect(const BBox& other) const;
//    BBox scale(const cv::Size& size) const;
//    BBox absolute(const cv::Size& size) const;
//    void print() const;
//};
//
//// Detection class declaration
//class Detection {
//public:
//    std::vector<cv::Point2f> keypoints;
//    float score;
//
//    Detection(const std::vector<cv::Point2f>& points, float score_val);
//    cv::Point2f operator[](size_t index) const;
//    size_t num_keypoints() const;
//    std::vector<cv::Point2f>::const_iterator begin() const;
//    std::vector<cv::Point2f>::const_iterator end() const;
//    BBox bbox() const;
//    Detection scaled(float factor) const;
//    void print() const;
//};
//
//#endif // TYPES_H





#ifndef TYPES_H
#define TYPES_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <cmath>
#include <optional>

// ImageTensor class declaration
class ImageTensor {
public:
    cv::Mat tensor_data;
    std::array<float, 4> padding;
    cv::Size original_size;

    ImageTensor(const cv::Mat& data, const std::array<float, 4>& pad, const cv::Size& original);
};

// Rect class declaration
class Rect {
public:
    float x_center, y_center, width, height, rotation;
    bool normalized;

    Rect(float x, float y, float w, float h, float r, bool norm);
    std::array<float, 2> size() const;
    Rect scaled(const std::array<float, 2>& size, bool normalize = false) const;
    std::vector<std::array<float, 2>> points() const;
};

// BBox class declaration
class BBox {
public:
    float xmin, ymin, xmax, ymax;

    BBox(float x1, float y1, float x2, float y2);
    std::array<float, 4> as_tuple() const;
    float width() const;
    float height() const;
    bool is_empty() const;
    bool normalized() const;
    float area() const;
    std::optional<BBox> intersect(const BBox& other) const;
    BBox scale(const cv::Size& size) const;
    BBox absolute(const cv::Size& size) const;
    void print() const;
};

// Detection class declaration
class Detection {
public:
    std::vector<cv::Point2f> keypoints;
    float score;

    Detection(const std::vector<cv::Point2f>& points, float score_val);
    cv::Point2f operator[](size_t index) const;
    size_t num_keypoints() const;
    std::vector<cv::Point2f>::const_iterator begin() const;
    std::vector<cv::Point2f>::const_iterator end() const;
    BBox bbox() const;
    Detection scaled(float factor) const;
    void print() const;
};

// struct Landmark {
//     float x, y, z;
//     Landmark(float x, float y, float z) : x(x), y(y), z(z) {}
// };

#endif // TYPES_H

