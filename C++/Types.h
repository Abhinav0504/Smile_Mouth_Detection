#ifndef TYPES_H
#define TYPES_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <cmath>


// ImageTensor class equivalent
class ImageTensor {
public:
    cv::Mat tensor_data;                    // Tensor data in a cv::Mat
    std::array<float, 4> padding;           // Padding array
    cv::Size original_size;                 // Original image size

    ImageTensor(const cv::Mat& data, const std::array<float, 4>& pad, const cv::Size& original)
            : tensor_data(data), padding(pad), original_size(original) {}
};



class Rect {
public:
    float x_center;
    float y_center;
    float width;
    float height;
    float rotation;
    bool normalized;

    // Constructor
    Rect(float x, float y, float w, float h, float r, bool norm)
            : x_center(x), y_center(y), width(w), height(h), rotation(r), normalized(norm) {}

    // Method to return the size of the rectangle
    std::array<float, 2> size() const {
        return {width, height};
    }

    // Method to scale the rectangle based on the given size
    Rect scaled(const std::array<float, 2>& size, bool normalize = false) const {
        if (normalized == normalize) return *this; // If already normalized, return self
        float sx = size[0];
        float sy = size[1];
        if (normalize) {
            sx = 1.0f / sx;
            sy = 1.0f / sy;
        }
        return Rect(x_center * sx, y_center * sy, width * sx, height * sy, rotation, false);
    }

    // Method to get the corner points of the rectangle considering rotation
    std::vector<std::array<float, 2>> points() const {
        float x = x_center;
        float y = y_center;
        float w = width / 2.0f;
        float h = height / 2.0f;

        // Four corners of the unrotated rectangle
        std::vector<std::array<float, 2>> pts = {
                {x - w, y - h},  // top-left
                {x + w, y - h},  // top-right
                {x + w, y + h},  // bottom-right
                {x - w, y + h}   // bottom-left
        };

        // If no rotation, return the points directly
        if (rotation == 0.0f) {
            return pts;
        }

        // Apply rotation around the center (x_center, y_center)
        float s = std::sin(rotation);
        float c = std::cos(rotation);
        std::vector<std::array<float, 2>> rotated_pts;

        for (const auto& pt : pts) {
            float dx = pt[0] - x;
            float dy = pt[1] - y;
            float new_x = x + (dx * c - dy * s);  // Rotated x
            float new_y = y + (dx * s + dy * c);  // Rotated y
            rotated_pts.push_back({new_x, new_y});
        }

        return rotated_pts;
    }
};



// BBox class equivalent
class BBox {
public:
    float xmin, ymin, xmax, ymax;

    BBox(float x1, float y1, float x2, float y2) : xmin(x1), ymin(y1), xmax(x2), ymax(y2) {}

    std::array<float, 4> as_tuple() const { return {xmin, ymin, xmax, ymax}; }

    float width() const { return xmax - xmin; }
    float height() const { return ymax - ymin; }
    bool empty() const { return width() <= 0 || height() <= 0; }
    bool normalized() const { return xmin >= -1 && xmax < 2 && ymin >= -1; }
    float area() const { return empty() ? 0.0f : width() * height(); }

    BBox intersect(const BBox& other) const {
        float new_xmin = std::max(xmin, other.xmin);
        float new_ymin = std::max(ymin, other.ymin);
        float new_xmax = std::min(xmax, other.xmax);
        float new_ymax = std::min(ymax, other.ymax);
        if (new_xmin < new_xmax && new_ymin < new_ymax) return BBox(new_xmin, new_ymin, new_xmax, new_ymax);
        return BBox(0, 0, 0, 0);  // No intersection
    }

    BBox scale(const cv::Size& size) const {
        return BBox(xmin * size.width, ymin * size.height, xmax * size.width, ymax * size.height);
    }

    BBox absolute(const cv::Size& size) const {
        return normalized() ? scale(size) : *this;
    }
};

// Landmark class equivalent
class Landmark {
public:
    float x, y, z;
    Landmark(float x_val, float y_val, float z_val) : x(x_val), y(y_val), z(z_val) {}
};

// Detection class equivalent to the Python version
class Detection {
public:
    // Vector of keypoints (2D points) representing [xmin, ymin, xmax, ymax, ...]
    std::vector<cv::Point2f> keypoints;
    // Confidence score for the detection
    float score;

    // Constructor: initialize keypoints and confidence score
    Detection(const std::vector<cv::Point2f>& points, float score_val)
            : keypoints(points), score(score_val) {}

    // Return the number of keypoints (excluding the bounding box points)
    int num_keypoints() const {
        return keypoints.size() - 2;  // Excluding xmin, ymin, xmax, ymax
    }

    // Access a keypoint by index (ignoring the first two points which are bbox)
    cv::Point2f operator[](size_t index) const {
        if (index + 2 >= keypoints.size()) {
            throw std::out_of_range("Index out of range for keypoints");
        }
        return keypoints[index + 2];  // Skipping the first two (bbox)
    }

    // Iterate over keypoints (ignores the bounding box)
    std::vector<cv::Point2f>::const_iterator begin() const {
        return keypoints.begin() + 2;
    }

    std::vector<cv::Point2f>::const_iterator end() const {
        return keypoints.end();
    }

    // Return the bounding box of this detection (xmin, ymin, xmax, ymax)
    BBox bbox() const {
        float xmin = keypoints[0].x;
        float ymin = keypoints[0].y;
        float xmax = keypoints[1].x;
        float ymax = keypoints[1].y;
        return BBox(xmin, ymin, xmax, ymax);
    }

    // Return a scaled version of this detection's keypoints and bounding box
    Detection scaled(float factor) const {
        std::vector<cv::Point2f> scaled_keypoints;
        // Scale all keypoints by the factor
        for (const auto& point : keypoints) {
            scaled_keypoints.emplace_back(point.x * factor, point.y * factor);
        }
        return Detection(scaled_keypoints, score);
    }
};

#endif // TYPES_H
