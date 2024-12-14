#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "FaceDetection.hpp"
#include "FaceLandmark.hpp"
#include "ModelLoader.hpp"

std::unique_ptr<tflite::Interpreter> LoadTFLiteModel(const std::string &model_path) {
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        throw std::runtime_error("Failed to load model: " + model_path);
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        throw std::runtime_error("Failed to build interpreter for model: " + model_path);
    }
    interpreter->AllocateTensors();
    return interpreter;
}

std::unique_ptr<tflite::Interpreter> LoadFaceDetectionModel(const std::string &model_path) {
    return LoadTFLiteModel(model_path);
}

std::unique_ptr<tflite::Interpreter> LoadFacialLandmarksModel(const std::string &model_path) {
    return LoadTFLiteModel(model_path);
}

std::unique_ptr<tflite::Interpreter> LoadKerasClassifierModel(const std::string &model_path) {
    return LoadTFLiteModel(model_path);
}

void RunInference(std::unique_ptr<tflite::Interpreter> &interpreter, const cv::Mat &input, int input_size) {
    float *input_data = interpreter->typed_input_tensor<float>(0);
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(input_size, input_size));
    resized.convertTo(resized, CV_32FC3, 1.0 / 255);
    std::memcpy(input_data, resized.data, resized.total() * resized.elemSize());
    interpreter->Invoke();
}


std::vector<cv::Point2f> GetLandmarks(std::unique_ptr<tflite::Interpreter> &interpreter) {
    float *output_data = interpreter->typed_output_tensor<float>(0);
    std::vector<cv::Point2f> landmarks;
    for (int i = 0; i < 124; ++i) {
        landmarks.emplace_back(output_data[2 * i] * 224, output_data[2 * i + 1] * 224);
    }
    return landmarks;
}


void DrawLandmarks(cv::Mat &image, const std::vector<cv::Point2f> &landmarks, const cv::Scalar &color, int thickness = 2) {
    for (const auto &point : landmarks) {
        cv::circle(image, point, 2, color, thickness);
    }
}

cv::Mat CropImage(const cv::Mat &image, const cv::Rect &rect) {
    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
    std::cout << "Bounding box: x=" << rect.x << ", y=" << rect.y
              << ", width=" << rect.width << ", height=" << rect.height << std::endl;
    // Ensure the bounding box does not exceed the image dimensions
    cv::Rect valid_rect = rect & cv::Rect(0, 0, image.cols, image.rows);

    if (valid_rect.width <= 0 || valid_rect.height <= 0) {
        throw std::runtime_error("Invalid bounding box. Cannot crop the image.");
    }

    return image(valid_rect);
//    return image(rect);
}

std::vector<cv::Point2f> GetMouthLandmarks(const std::vector<cv::Point2f> &landmarks) {
    std::vector<cv::Point2f> mouth_landmarks;
    for (int i = 48; i < 68; ++i) {
        mouth_landmarks.push_back(landmarks[i]);
    }
    return mouth_landmarks;
}

cv::Rect CalculateBoundingBox(const std::vector<cv::Point2f> &landmarks) {
    float min_x = landmarks[0].x;
    float min_y = landmarks[0].y;
    float max_x = landmarks[0].x;
    float max_y = landmarks[0].y;

    for (const auto &landmark : landmarks) {
        if (landmark.x < min_x) min_x = landmark.x;
        if (landmark.y < min_y) min_y = landmark.y;
        if (landmark.x > max_x) max_x = landmark.x;
        if (landmark.y > max_y) max_y = landmark.y;
    }

    return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

cv::Rect ConvertToBoundingBoxUsingWidthHeight(float *bbox, int image_width, int image_height) {
    // Use the bounding box format [x, y, width, height]
    int x = static_cast<int>(bbox[0] * image_width);   // Top-left x coordinate
    int y = static_cast<int>(bbox[1] * image_height);  // Top-left y coordinate
    int width = static_cast<int>(bbox[2] * image_width); // Width
    int height = static_cast<int>(bbox[3] * image_height); // Height

    // Ensure the bounding box does not exceed the image boundaries
    if (x < 0) x = 0;
//    x = 0;
    if (y < 0) y = 0;
    if (x + width > image_width) width = image_width - x;
    if (y + height > image_height) height = image_height - y;

    return cv::Rect(x, y, width, height);
}


std::string ClassifySmile(std::unique_ptr<tflite::Interpreter> &interpreter, const cv::Mat &face_image, const std::vector<cv::Point2f> &landmarks, bool use_mouth_landmarks) {
    int input_size = 64; // model input size is 64x64
    float *classification_input = interpreter->typed_input_tensor<float>(0);
    cv::Mat resized_face;
    cv::resize(face_image, resized_face, cv::Size(input_size, input_size));
    resized_face.convertTo(resized_face, CV_32FC3, 1.0 / 255);
    std::memcpy(classification_input, resized_face.data, resized_face.total() * resized_face.elemSize());

    std::vector<cv::Point2f> input_landmarks = landmarks;
    if (use_mouth_landmarks) {
        input_landmarks = GetMouthLandmarks(landmarks);
    }

    float *landmarks_input = interpreter->typed_input_tensor<float>(1);
    for (size_t i = 0; i < input_landmarks.size(); ++i) {
        landmarks_input[2 * i] = input_landmarks[i].x / 224.0f;
        landmarks_input[2 * i + 1] = input_landmarks[i].y / 224.0f;
    }

    interpreter->Invoke();
    float *classification_output = interpreter->typed_output_tensor<float>(0);
    float classification_score = classification_output[1]; // the second class is "smile"

    return (classification_score > 0.5) ? "Smile" : "No Smile";
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path> [--mouth-only]" << std::endl; // to read only the cropped image from mouth landmarks
        return -1;
    }

    std::string image_path = argv[1];
    bool use_mouth_landmarks = false;

    if (argc > 2 && std::string(argv[2]) == "--mouth-only") {
        use_mouth_landmarks = true;
    }


    auto face_detection_model = LoadFaceDetectionModel("/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/Final_code/models/face_detection_short.tflite"); // path to face detection model
//    Check if face_detection_model is a Valid Pointer:
    if (!face_detection_model) {
        std::cerr << "Error: face_detection_model is not initialized!" << std::endl;
        return -1;
    }

    auto facial_landmarks_model = LoadFacialLandmarksModel("/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/Final_code/models/TensorFlowFacialLandmarksV1.tflite"); // path to face landmarks model
    auto keras_classification_model = LoadKerasClassifierModel("/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/Final_code/models/best_model_whole_face_124_lm_attention.tflite"); // path to smile/mouth model

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }


    RunInference(face_detection_model, image, 224);
    float *bbox = face_detection_model->typed_output_tensor<float>(0);
    std::cout << "Raw bounding box values: " << bbox[0] << ", " << bbox[1] << ", " << bbox[2] << ", " << bbox[3] << std::endl;

    // Convert to absolute pixel values using [x, y, width, height] format
    cv::Rect face_rect = ConvertToBoundingBoxUsingWidthHeight(bbox, image.cols, image.rows);

    cv::Mat face_image = image(face_rect);

    // Draw the bounding box
    cv::rectangle(image, face_rect, cv::Scalar(0, 255, 0), 2);

    // Run inference on facial landmarks model
    RunInference(facial_landmarks_model, face_image, 224);
    std::vector<cv::Point2f> landmarks = GetLandmarks(facial_landmarks_model);

    // Scale landmarks back to the main image size
    for (auto &point : landmarks) {
        point.x = face_rect.x + point.x * face_rect.width;
        point.y = face_rect.y + point.y * face_rect.height;
    }

    // Draw the landmarks on the main image
    DrawLandmarks(image, landmarks, cv::Scalar(255, 0, 0));


    // Display the image with the bounding box
    cv::imshow("Face Detection Results", image);
    cv::waitKey(0);





//    std::cout << "Raw bounding box values: " << bbox[0] << ", " << bbox[1] << ", " << bbox[2] << ", " << bbox[3] << std::endl;
//
//    // Verify if the bounding box values are normalized (between 0 and 1)
//    if (bbox[0] < 0 || bbox[1] < 0 || bbox[2] > 1 || bbox[3] > 1) {
//        std::cerr << "Warning: Bounding box values are not normalized! Check the model output format." << std::endl;
//    }
//
//    // Use the correct format based on output interpretation
//    cv::Rect face_rect;
//    if (bbox[2] <= 1 && bbox[3] <= 1) { // Normalized format: [x_min, y_min, x_max, y_max]
//        face_rect = cv::Rect(bbox[0] * image.cols, bbox[1] * image.rows,
//                             (bbox[2] - bbox[0]) * image.cols, (bbox[3] - bbox[1]) * image.rows);
//    } else { // Absolute format: [x, y, width, height]
//        face_rect = cv::Rect(bbox[0], bbox[1], bbox[2], bbox[3]);
//    }
//
//    // Draw and display the bounding box
//    cv::rectangle(image, face_rect, cv::Scalar(0, 255, 0), 2);
//    cv::imshow("Face Detection Result", image);
//    cv::waitKey(0);









////    cv::Rect face_rect(bbox[0] * image.cols, bbox[1] * image.rows, bbox[2] * image.cols, bbox[3] * image.rows);
////    cv::Mat face_image = CropImage(image, face_rect);
//
//    // Calculate the bounding box in pixel coordinates
//    cv::Rect face_rect(bbox[0] * image.cols, bbox[1] * image.rows,
//                       (bbox[2] - bbox[0]) * image.cols, (bbox[3] - bbox[1]) * image.rows);
//
//    // Draw the bounding box on the image
//    cv::rectangle(image, face_rect, cv::Scalar(0, 255, 0), 2);  // Green bounding box with thickness 2
//
//    // Display the image with the bounding box
//    cv::imshow("Face Detection Result", image);
//    cv::waitKey(0);  // Wait indefinitely until a key is pressed
//
////    RunInference(facial_landmarks_model, face_image, 224);
////    std::vector<cv::Point2f> landmarks = GetLandmarks(facial_landmarks_model);
//
////    std::string result = ClassifySmile(keras_classification_model, face_image, landmarks, use_mouth_landmarks);
////    std::cout << "Result: " << result << std::endl;

    return 0;
}
