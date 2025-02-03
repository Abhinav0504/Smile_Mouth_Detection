#include "FaceLandmark.hpp"
#include <iostream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include "Config.h"
#include <tensorflow/lite/model.h>

FaceLandmark::FaceLandmark(const std::string& modelPath) {
    std::cout << "DEBUG: am6176_land.cpp is being used." << std::endl;

    // Use default model path if none is provided
    std::string resolvedPath = modelPath.empty() ? Config::model_land : modelPath;

    // Load the TensorFlow Lite model
    auto model = tflite::FlatBufferModel::BuildFromFile(resolvedPath.c_str());
    if (!model) {
        throw std::runtime_error("Failed to load the model from path: " + resolvedPath);
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        throw std::runtime_error("Failed to create interpreter for model.");
    }

    // Get input and output details
    inputIndex = interpreter->inputs()[0];
    inputShape = std::vector<int>(
            interpreter->tensor(inputIndex)->dims->data,
            interpreter->tensor(inputIndex)->dims->data + interpreter->tensor(inputIndex)->dims->size
    );
    dataIndex = interpreter->outputs()[0];
    faceIndex = interpreter->outputs()[1];

    // Validate model output size
    int numExpectedElements = NUM_DIMS * NUM_LANDMARKS;
    if (interpreter->tensor(dataIndex)->dims->data[interpreter->tensor(dataIndex)->dims->size - 1] < numExpectedElements) {
        throw std::runtime_error("Incompatible model: output size is less than expected.");
    }

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Failed to allocate tensors.");
    }
}

std::vector<cv::Point3f> FaceLandmark::operator()(const cv::Mat& image, const cv::Rect2f& roi) {
    std::cout << "DEBUG: Placeholder for operator()" << std::endl;
    return std::vector<cv::Point3f>(NUM_LANDMARKS, cv::Point3f(0.0f, 0.0f, 0.0f));
}

cv::Rect2f face_detection_to_roi(const Detection& face_detection, const cv::Size& image_size) {
    std::cout << "DEBUG: face_detection_to_roi()" << std::endl;
    return cv::Rect2f(0.1f, 0.1f, 0.8f, 0.8f); // Dummy return value
}

std::vector<cv::Point3f> project_landmarks(const cv::Mat& raw_data, const cv::Size& tensor_size,
                                           const cv::Size& image_size, const cv::Scalar& padding,
                                           const cv::Rect2f& roi) {
    std::cout << "DEBUG: project_landmarks()" << std::endl;
    return std::vector<cv::Point3f>(NUM_LANDMARKS, cv::Point3f(0.0f, 0.0f, 0.0f)); // Dummy return
}

std::vector<Annotation> face_landmarks_to_render_data(
        const std::vector<cv::Point3f>& face_landmarks,
        const Color& landmark_color,
        const Color& connection_color,
        float thickness,
        std::vector<Annotation>* output) {
    std::cout << "DEBUG: face_landmarks_to_render_data()" << std::endl;
    return {}; // Dummy return
}
