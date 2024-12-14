// transforms.h
#ifndef TRANSFORMS_H
#define TRANSFORMS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include "Types.h"


cv::Mat _normalize_image(const cv::Mat& image);
cv::Mat _perspective_transform_coeff(const std::vector<cv::Point2f>& src_points, const std::vector<cv::Point2f>& dst_points);
ImageTensor image_to_tensor(const cv::Mat& image, Rect& roi, cv::Size output_size, bool keep_aspect_ratio, std::pair<float, float> output_range, bool flip_horizontal);
cv::Rect convert_to_cv_rect(const Rect& roi, const cv::Size& image_size);
std::vector<Detection> detection_letterbox_removal(
        const std::vector<Detection>& detections,
        const std::array<float, 4>& padding);


#endif // TRANSFORMS_H
