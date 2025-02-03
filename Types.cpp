#include "Types.h"
#include <iostream>

// ImageTensor constructor
ImageTensor::ImageTensor(const cv::Mat& data, const std::array<float, 4>& pad, const cv::Size& original)
        : tensor_data(data), padding(pad), original_size(original) {}

// Rect constructor
Rect::Rect(float x, float y, float w, float h, float r, bool norm)
        : x_center(x), y_center(y), width(w), height(h), rotation(r), normalized(norm) {}

std::array<float, 2> Rect::size() const {
    return {width, height};
}

Rect Rect::scaled(const std::array<float, 2>& size, bool normalize) const {
    if (normalized == normalize) return *this;
    float sx = size[0];
    float sy = size[1];
    if (normalize) {
        sx = 1.0f / sx;
        sy = 1.0f / sy;
    }
    return Rect(x_center * sx, y_center * sy, width * sx, height * sy, rotation, false);
}

std::vector<std::array<float, 2>> Rect::points() const {
    float x = x_center;
    float y = y_center;
    float w = width / 2.0f;
    float h = height / 2.0f;

    std::vector<std::array<float, 2>> pts = {
            {x - w, y - h}, {x + w, y - h}, {x + w, y + h}, {x - w, y + h}
    };

    if (rotation == 0.0f) return pts;

    float s = std::sin(rotation);
    float c = std::cos(rotation);
    std::vector<std::array<float, 2>> rotated_pts;
    for (const auto& pt : pts) {
        float dx = pt[0] - x;
        float dy = pt[1] - y;
        rotated_pts.push_back({x + dx * c - dy * s, y + dx * s + dy * c});
    }
    return rotated_pts;
}




// BBox methods
BBox::BBox(float x1, float y1, float x2, float y2) : xmin(x1), ymin(y1), xmax(x2), ymax(y2) {}

std::array<float, 4> BBox::as_tuple() const { return {xmin, ymin, xmax, ymax}; }
float BBox::width() const { return xmax - xmin; }
float BBox::height() const { return ymax - ymin; }
bool BBox::is_empty() const { return width() <= 0 || height() <= 0; }
bool BBox::normalized() const { return xmin >= -1 && xmax < 2 && ymin >= -1; }
float BBox::area() const { return is_empty() ? 0.0f : width() * height(); }

std::optional<BBox> BBox::intersect(const BBox& other) const {
    float new_xmin = std::max(xmin, other.xmin);
    float new_ymin = std::max(ymin, other.ymin);
    float new_xmax = std::min(xmax, other.xmax);
    float new_ymax = std::min(ymax, other.ymax);

    if (new_xmin < new_xmax && new_ymin < new_ymax) {
        return BBox(new_xmin, new_ymin, new_xmax, new_ymax);
    }
    return std::nullopt;
}

BBox BBox::scale(const cv::Size& size) const {
    return BBox(xmin * size.width, ymin * size.height, xmax * size.width, ymax * size.height);
}

BBox BBox::absolute(const cv::Size& size) const {
    return normalized() ? scale(size) : *this;
}

void BBox::print() const {
    std::cout << "BBox(xmin=" << xmin << ", ymin=" << ymin
              << ", xmax=" << xmax << ", ymax=" << ymax << ")" << std::endl;
}




// Detection methods
Detection::Detection(const std::vector<cv::Point2f>& points, float score_val)
        : keypoints(points), score(score_val) {}

cv::Point2f Detection::operator[](size_t index) const {
    if (index + 2 >= keypoints.size()) throw std::out_of_range("Index out of range for keypoints");
    return keypoints[index + 2];
}

size_t Detection::num_keypoints() const {
    return keypoints.size() > 2 ? keypoints.size() - 2 : 0;
}

std::vector<cv::Point2f>::const_iterator Detection::begin() const { return keypoints.begin() + 2; }
std::vector<cv::Point2f>::const_iterator Detection::end() const { return keypoints.end(); }

BBox Detection::bbox() const {
    if (keypoints.size() < 2) throw std::runtime_error("Insufficient keypoints for bounding box");
    return BBox(keypoints[0].x, keypoints[0].y, keypoints[1].x, keypoints[1].y);
}

Detection Detection::scaled(float factor) const {
    std::vector<cv::Point2f> scaled_keypoints;
    for (const auto& point : keypoints) scaled_keypoints.emplace_back(point.x * factor, point.y * factor);
    return Detection(scaled_keypoints, score);
}

void Detection::print() const {
    std::cout << "Bounding Box: [" << keypoints[0].x << ", " << keypoints[0].y
              << ", " << keypoints[1].x << ", " << keypoints[1].y << "], Score: " << score << std::endl;

    if (num_keypoints() > 0) {
        std::cout << "Keypoints: ";
        for (auto it = begin(); it != end(); ++it) {
            std::cout << "[" << it->x << ", " << it->y << "] ";
        }
        std::cout << std::endl;
    }
}
