#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "FaceDetection_am6176.h"

// Function to load a TensorFlow Lite model using TensorFlow Lite C++ API
std::unique_ptr<tflite::Interpreter> LoadModel(const std::string& model_path) {
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return nullptr;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to construct interpreter." << std::endl;
        return nullptr;
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors!" << std::endl;
        return nullptr;
    }

    return interpreter;
}

// Function to "predict" based on the image, model, classifier, and landmarks
std::tuple<cv::Mat, cv::Mat, int, float> predict(cv::Mat& image, std::unique_ptr<tflite::Interpreter>& model, const std::string& classifier, int landmarks_count) {

    std::cout << std::endl << "STEP 1: Load in the Image" << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "Original Image Shape: " << image.rows << " x " << image.cols << " x " << image.channels() << std::endl;
    cv::Mat image_rgb;
    cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
    std::cout << "Converted to RGB Image Shape: " << image_rgb.rows << " x " << image_rgb.cols << " x " << image_rgb.channels() << std::endl;

    // Print the first 10 and last 10 RGB pixel values
    std::cout << "\nFirst 10 RGB values after BGR to RGB conversion:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        cv::Vec3b pixel = image_rgb.at<cv::Vec3b>(0, i);
        std::cout << "[" << static_cast<int>(pixel[0]) << ", "
                  << static_cast<int>(pixel[1]) << ", "
                  << static_cast<int>(pixel[2]) << "]" << std::endl;
    }

    std::cout << "\nLast 10 RGB values after BGR to RGB conversion:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        cv::Vec3b pixel = image_rgb.at<cv::Vec3b>(image_rgb.rows - 1, image_rgb.cols - i - 1);
        std::cout << "[" << static_cast<int>(pixel[0]) << ", "
                  << static_cast<int>(pixel[1]) << ", "
                  << static_cast<int>(pixel[2]) << "]" << std::endl;
    }


    std::cout << "Image size (equivalent to PIL Image): " << image_rgb.cols << " x " << image_rgb.rows << std::endl;

    try {
        // Create FaceDetection instance using FRONT_CAMERA model
        FaceDetection face_detector(FaceDetection::FaceDetectionModel::FRONT_CAMERA);
        face_detector.printInterpreterInfo();

        // Run the face detection model
        std::vector<Detection> detections = face_detector(image_rgb);
        // Print the number of faces detected
        std::cout << "Number of faces detected: " << detections.size() << std::endl;



    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }





    // Dummy operation to simulate prediction (replace this with actual inference logic)
    cv::Mat cropped_image = image; // Dummy cropped image
    cv::Mat output = image.clone(); // Dummy output image

    // Return dummy values for result and score (to simulate classification)
    int result = (classifier == "M") ? 1 : 0;  // Example: 1 for mouth, 0 for smile
    float score = 0.75f;  // Dummy confidence score

    return std::make_tuple(output, cropped_image, result, score);
}

// Function based on Python's `run_single_image` function
void run_single_image(std::string classifier, int landmarks_count, cv::Mat& image, std::unique_ptr<tflite::Interpreter>& model, float threshold) {
    // Call the predict function (similar to how Python calls predict in run_single_image)
    auto [output, cropped_image, result, score] = predict(image, model, classifier, landmarks_count);

    // Dummy logic to classify based on score and classifier
    std::string classification;
    if (classifier == "M" && score > threshold) {
        classification = "Mouth-close";
    } else if (classifier == "M" && score <= threshold) {
        classification = "Mouth-open";
    } else if (classifier == "S" && score >= threshold) {
        classification = "No-smile";
    } else {
        classification = "Smile";
    }

    classification += " : " + std::to_string(score);

    // Display the result on the image
    cv::putText(output, classification, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    cv::imshow("Result", output);
    cv::waitKey(0);
}

int main(int argc, char** argv) {
    std::string classifier = "M";  // Default is Mouth classifier
    std::string image_path;
    std::string model_path = "/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/Final_code/models/face_detection_short.tflite";  // Specify the default model path
    int landmarks_count = 40;      // Default number of landmarks
    float threshold = 0.5;         // Default classification threshold

    // Command-line argument parsing
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--classifier") {
            classifier = argv[++i];
        } else if (std::string(argv[i]) == "--image") {
            image_path = argv[++i];
        } else if (std::string(argv[i]) == "--model") {
            model_path = argv[++i];
        } else if (std::string(argv[i]) == "--landmark") {
            landmarks_count = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--threshold") {
            threshold = std::stof(argv[++i]);
        }
    }

    if (image_path.empty()) {
        std::cerr << "Please specify an image path using --image" << std::endl;
        return -1;
    }

    auto interpreter = LoadModel(model_path);
    if (!interpreter) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }

    // Load the input image
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);


    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    // Call run_single_image function
    run_single_image(classifier, landmarks_count, image, interpreter, threshold);

    return 0;
}
