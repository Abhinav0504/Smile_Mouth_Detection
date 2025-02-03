//#include "render.h"
//
//// Define Colors
//const Color Colors::BLACK = {0, 0, 0, -1};
//const Color Colors::RED = {255, 0, 0, -1};
//const Color Colors::GREEN = {0, 255, 0, -1};
//const Color Colors::BLUE = {0, 0, 255, -1};
//const Color Colors::PINK = {255, 0, 255, -1};
//const Color Colors::WHITE = {255, 255, 255, -1};
//
//// Implement Color::asScalar
//cv::Scalar Color::asScalar() const {
//    return (a >= 0) ? cv::Scalar(b, g, r, a) : cv::Scalar(b, g, r);
//}
//
//// Implement Point::asCVPoint
//cv::Point2f Point::asCVPoint() const {
//    return cv::Point2f(x, y);
//}
//
//// Implement RectOrOval::asCVRect
//cv::Rect RectOrOval::asCVRect() const {
//    return cv::Rect(cv::Point(left, top), cv::Point(right, bottom));
//}
//
//// Convert detections into annotations
//std::vector<Annotation> detections_to_render_data(
//        const std::vector<Detection>& detections,
//        const Color& bounds_color,
//        const Color& keypoint_color,
//        int line_width,
//        int point_width,
//        bool normalized_positions) {
//
//    std::vector<Annotation> annotations;
//
//    // Add bounding box annotations
//    if (bounds_color.r >= 0) {
//        for (const auto& detection : detections) {
//            Annotation annotation;
//
//            // Convert detection.bbox() to RectOrOval
//            auto bbox = detection.bbox();  // Call the bbox() function to get the bounding box
//            annotation.rect = {
//                    bbox.xmin, bbox.ymin,
//                    bbox.xmax, bbox.ymax,
//                    false  // Assuming not an oval
//            };
//
//
//            annotation.color = bounds_color;
//            annotation.normalized_positions = normalized_positions;
//            annotation.thickness = line_width;
//            annotations.push_back(annotation);
//        }
//    }
//
//    // Add keypoint annotations
//    if (keypoint_color.r >= 0) {
//        for (const auto& detection : detections) {
//            Annotation annotation;
//
//            // Convert detection.keypoints (cv::Point2f) to Point
//            annotation.points.reserve(detection.keypoints.size());
//            for (const auto& keypoint : detection.keypoints) {
//                annotation.points.push_back({keypoint.x, keypoint.y});
//            }
//
//            annotation.color = keypoint_color;
//            annotation.normalized_positions = normalized_positions;
//            annotation.thickness = point_width;
//            annotations.push_back(annotation);
//        }
//    }
//
//    return annotations;
//}
//
//// Render annotations on an image
//cv::Mat render_to_image(const std::vector<Annotation>& annotations, cv::Mat& image) {
//    for (const auto& annotation : annotations) {
//        // Draw bounding box
//        if (annotation.rect.left != 0 || annotation.rect.top != 0 ||
//            annotation.rect.right != 0 || annotation.rect.bottom != 0) {
//            cv::rectangle(image, annotation.rect.asCVRect(),
//                          annotation.color.asScalar(), annotation.thickness);
//        }
//
//        // Draw keypoints
//        for (const auto& point : annotation.points) {
//            cv::circle(image, point.asCVPoint(), annotation.thickness,
//                       annotation.color.asScalar(), cv::FILLED);
//        }
//    }
//
//    return image;
//}


#include "render.h"

// Define Colors
const Color Colors::BLACK = {0, 0, 0, -1};
const Color Colors::RED = {255, 0, 0, -1};
const Color Colors::GREEN = {0, 255, 0, -1};
const Color Colors::BLUE = {0, 0, 255, -1};
const Color Colors::PINK = {255, 0, 255, -1};
const Color Colors::WHITE = {255, 255, 255, -1};

// Implement Color::asScalar
cv::Scalar Color::asScalar() const {
    return (a >= 0) ? cv::Scalar(b, g, r, a) : cv::Scalar(b, g, r);
}

// Implement Point::asCVPoint
cv::Point2f Point::asCVPoint() const {
    return cv::Point2f(x, y);
}

// Implement RectOrOval::asCVRect
cv::Rect RectOrOval::asCVRect() const {
    int n_left = static_cast<int>(left * 500);
    int n_top = static_cast<int>(top * 500);
    int n_right = static_cast<int>(right * 500);
    int n_bottom = static_cast<int>(bottom * 500);

    return cv::Rect(cv::Point(n_left, n_top), cv::Point(n_right, n_bottom));
}

// Convert detections into annotations
std::vector<Annotation> detections_to_render_data(
        const std::vector<Detection>& detections,
        const Color& bounds_color,
        const Color& keypoint_color,
        int line_width,
        int point_width,
        bool normalized_positions) {

    std::vector<Annotation> annotations;

    // Add bounding box annotations
    if (bounds_color.g >= 0) {
        for (const auto& detection : detections) {
            Annotation annotation;

            // Convert detection.bbox() to RectOrOval
            auto bbox = detection.bbox();  // Call the bbox() function to get the bounding box
            annotation.rect = {
                    bbox.xmin, bbox.ymin,
                    bbox.xmax, bbox.ymax,
                    false  // Assuming not an oval
            };


            annotation.color = bounds_color;
            annotation.normalized_positions = normalized_positions;
            annotation.thickness = line_width;
            annotations.push_back(annotation);
        }
    }

    // Add keypoint annotations
    if (keypoint_color.g >= 0) {
        for (const auto& detection : detections) {
            Annotation annotation;

            // Convert detection.keypoints (cv::Point2f) to Point
            annotation.points.reserve(detection.keypoints.size());
            for (const auto& keypoint : detection.keypoints) {
                annotation.points.push_back({keypoint.x, keypoint.y});
            }

            annotation.color = keypoint_color;
            annotation.normalized_positions = normalized_positions;
            annotation.thickness = point_width;
            annotations.push_back(annotation);
        }
    }

    return annotations;
}

// Render annotations on an image
cv::Mat render_to_image(const std::vector<Annotation>& annotations, cv::Mat& image) {
    for (const auto& annotation : annotations) {
        // Draw bounding box
        if (annotation.rect.left != 0 || annotation.rect.top != 0 ||
            annotation.rect.right != 0 || annotation.rect.bottom != 0) {
            cv::rectangle(image, annotation.rect.asCVRect(),
                          annotation.color.asScalar(), annotation.thickness);
        }

        // Draw keypoints
        for (const auto& point : annotation.points) {
            cv::circle(image, point.asCVPoint(), annotation.thickness,
                       annotation.color.asScalar(), cv::FILLED);
        }
    }

    return image;
}

