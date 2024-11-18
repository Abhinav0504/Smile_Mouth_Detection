#ifndef FACE_DETECTION_HPP
#define FACE_DETECTION_HPP

#include <string>
#include <vector>
#include <memory>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include "Config.h"
#include "Types.h"
#include "Transforms.h"


// SSDOptions Struct for anchor generation
struct SSDOptions {
    int num_layers;
    int input_size_height;
    int input_size_width;
    float anchor_offset_x;
    float anchor_offset_y;
    std::vector<int> strides;
    float interpolated_scale_aspect_ratio;
};

class FaceDetection {
public:
    // Enum for model types
    enum class FaceDetectionModel {
        FRONT_CAMERA = 0,
        BACK_CAMERA,
        SHORT,
        FULL,
        FULL_SPARSE
    };

    // Constructor: initializes model_path, SSD options, and TFLite interpreter
    FaceDetection(FaceDetectionModel model_type = FaceDetectionModel::FRONT_CAMERA, const std::string& model_path = "");

    // Function to generate SSD anchors (equivalent to _ssd_generate_anchors)
    std::vector<std::pair<float, float>> generate_anchors(const SSDOptions& options);

    // Function to print input/output indices for debugging
    void printInterpreterInfo() const;

    // Overload operator() to make the object callable like in Python
    std::vector<Detection> operator()(const cv::Mat& image, Rect* roi = nullptr);

    // New method declarations
    std::vector<cv::Rect> decode_boxes(const float* raw_boxes); // Decodes raw bounding boxes
    std::vector<float> get_sigmoid_scores(const float* raw_scores); // Applies sigmoid to raw scores
    std::vector<Detection> convert_to_detections(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores); // Converts boxes and scores into Detection objects


    // Getter functions
    std::vector<std::pair<float, float>> get_anchors() const { return anchors; }

private:
    std::string model_path;
    std::vector<std::pair<float, float>> anchors;  // List of anchor points
    std::unique_ptr<tflite::Interpreter> interpreter;
    int input_index;
    std::vector<int> input_shape;
    int bbox_index;
    int score_index;
};

#endif // FACE_DETECTION_HPP
