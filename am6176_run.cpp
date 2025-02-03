//#include <iostream>
//#include <vector>
//#include <memory>
//#include <opencv2/opencv.hpp>
//#include "tensorflow/lite/interpreter.h"
//#include "tensorflow/lite/kernels/register.h"
//#include "tensorflow/lite/model.h"
//#include "tensorflow/lite/builtin_op_data.h"
//#include "FaceDetection_am6176.h"
//#include "render.h"
//
//
//
//// Function to load a TensorFlow Lite model using Tensorgit branch --set-upstream-to=github/Smile_Mouth_TF_AbhinavFlow Lite C++ API
//std::unique_ptr<tflite::Interpreter> LoadModel(const std::string& model_path) {
//    auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
//    if (!model) {
//        std::cerr << "Failed to load model: " << model_path << std::endl;
//        return nullptr;
//    }
//
//    tflite::ops::builtin::BuiltinOpResolver resolver;
//    std::unique_ptr<tflite::Interpreter> interpreter;
//    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
//    if (!interpreter) {
//        std::cerr << "Failed to construct interpreter." << std::endl;
//        return nullptr;
//    }
//
//    if (interpreter->AllocateTensors() != kTfLiteOk) {
//        std::cerr << "Failed to allocate tensors!" << std::endl;
//        return nullptr;
//    }
//
//    return interpreter;
//}
//
//cv::Mat cropFace(const cv::Mat& image, const std::vector<std::vector<float>>& renderData) {
//    /**
//     * This method crops the face based on the provided render data.
//     */
//
//    // Make a copy of the original image
//    cv::Mat orig = image.clone();
//    int imageWidth = image.cols;
//    int imageHeight = image.rows;
//
//    // Normalized coordinates of the bounding box
//    float left = renderData[0][0];   // Assuming left is at index 0
//    float top = renderData[0][1];    // Assuming top is at index 1
//    float right = renderData[0][2];  // Assuming right is at index 2
//    float bottom = renderData[0][3]; // Assuming bottom is at index 3
//
//    // Calculating the actual pixel values of the bounding box
//    int actualLeft = static_cast<int>(left * imageWidth);
//    int actualTop = static_cast<int>(top * imageHeight);
//    int actualRight = static_cast<int>(right * imageWidth);
//    int actualBottom = static_cast<int>(bottom * imageHeight);
//
//    // Extend the bounding box by 100 pixels on all sides
//    int x1 = std::max(0, actualLeft - 100);
//    int y1 = std::max(0, actualTop - 100);
//    int x2 = std::min(imageWidth, actualRight + 100);
//    int y2 = std::min(imageHeight, actualBottom + 100);
//
//    // Crop the image
//    cv::Rect cropRegion(x1, y1, x2 - x1, y2 - y1);
//    cv::Mat cropped = orig(cropRegion);
//
//    // Resize the cropped image to 512x512
//    cv::resize(cropped, cropped, cv::Size(512, 512));
//
//    return cropped;
//}
//
//
//// Function to "predict" based on the image, model, classifier, and landmarks
//std::tuple<cv::Mat, cv::Mat, int, float> predict(cv::Mat& image, std::unique_ptr<tflite::Interpreter>& model, const std::string& classifier, int landmarks_count) {
//
//    std::cout << std::endl << "STEP 1: Load in the Image" << std::endl;
//    std::cout << "---------------------------" << std::endl;
//    std::cout << "Original Image Shape: " << image.rows << " x " << image.cols << " x " << image.channels() << std::endl;
//
////    cv::imwrite("original_image.png", image);
//
//
//    cv::Mat image_rgb;
//    cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
//    std::cout << "Converted to RGB Image Shape: " << image_rgb.rows << " x " << image_rgb.cols << " x " << image_rgb.channels() << std::endl;
//
////    cv::imwrite("BGRtoRGB_image.png", image_rgb);
//
//
//    // Print the first 10 and last 10 RGB pixel values
//    std::cout << "\nFirst 10 RGB values after BGR to RGB conversion:" << std::endl;
//    for (int i = 0; i < 10; ++i) {
//        cv::Vec3b pixel = image_rgb.at<cv::Vec3b>(0, i);
//        std::cout << "[" << static_cast<int>(pixel[0]) << ", "
//                  << static_cast<int>(pixel[1]) << ", "
//                  << static_cast<int>(pixel[2]) << "]" << std::endl;
//    }
//
//    std::cout << "\nLast 10 RGB values after BGR to RGB conversion:" << std::endl;
//    for (int i = 0; i < 10; ++i) {
//        cv::Vec3b pixel = image_rgb.at<cv::Vec3b>(image_rgb.rows - 1, image_rgb.cols - i - 1);
//        std::cout << "[" << static_cast<int>(pixel[0]) << ", "
//                  << static_cast<int>(pixel[1]) << ", "
//                  << static_cast<int>(pixel[2]) << "]" << std::endl;
//    }
//
//
//    std::cout << "Image size (equivalent to PIL Image): " << image_rgb.cols << " x " << image_rgb.rows << std::endl;
//
//    cv::Mat cropped_image = cv::Mat::zeros(1, 1, CV_8UC3); // Dummy empty image
//
//
//    try {
//        // Create FaceDetection instance using FRONT_CAMERA model
//        FaceDetection face_detector(FaceDetection::FaceDetectionModel::FRONT_CAMERA);
//        face_detector.printInterpreterInfo();
//
//        // Run the face detection model
//        std::vector<Detection> detections = face_detector(image_rgb);
//        // Print the number of faces detected
////        std::cout << "Number of faces detected: " << detections.size() << std::endl;
//
//        // Generate render data for bounding boxes
//        std::vector<Annotation> render_data = detections_to_render_data(
//                detections, Colors::GREEN, Colors::BLUE, 2, 4, true);
//
//        // Debugging: Print render data
//        std::cout << std::endl << "Render Data:" << std::endl;
//        for (const auto& annotation : render_data) {
//            std::cout << "  Color: (" << annotation.color.r << ", "
//                      << annotation.color.g << ", " << annotation.color.b << ")" << std::endl;
//            std::cout << "  Thickness: " << annotation.thickness << std::endl;
//            std::cout << "  Normalized Positions: " << (annotation.normalized_positions ? "True" : "False") << std::endl;
//
//            for (const auto& point : annotation.points) {
//                std::cout << "    Keypoint: (" << point.x << ", " << point.y << ")" << std::endl;
//            }
//
//            std::cout << "    Bounding Box: ("
//                      << annotation.rect.left << ", " << annotation.rect.top << ", "
//                      << annotation.rect.right << ", " << annotation.rect.bottom << ")"
//                      << std::endl;
//        }
//
//        // Render annotations onto the image
//        cv::Mat annotated_image = render_to_image(render_data, image_rgb);
//        // Show the annotated image
//        cv::imshow("Annotated Image", annotated_image);
//        cv::waitKey(0);
//        cv::imwrite("annotated.jpg", annotated_image);
////        cv::Mat cropped_image = cropFace(image, render_data);
//
//    } catch (const std::exception &e) {
//        std::cerr << "Error: " << e.what() << std::endl;
//    }
//    cv::Mat output = image.clone(); // Dummy output image
//
//    // Return dummy values for result and score (to simulate classification)
//    int result = (classifier == "M") ? 1 : 0;  // Example: 1 for mouth, 0 for smile
//    float score = 0.75f;  // Dummy confidence score
//
//    return std::make_tuple(output, cropped_image, result, score);
//}
//
//
//
//// Function based on Python's `run_single_image` function
//void run_single_image(std::string classifier, int landmarks_count, cv::Mat& image, std::unique_ptr<tflite::Interpreter>& model, float threshold) {
//    // Call the predict function (similar to how Python calls predict in run_single_image)
//    auto [output, cropped_image, result, score] = predict(image, model, classifier, landmarks_count);
//
//    // Dummy logic to classify based on score and classifier
//    std::string classification;
//    if (classifier == "M" && score > threshold) {
//        classification = "Mouth-close";
//    } else if (classifier == "M" && score <= threshold) {
//        classification = "Mouth-open";
//    } else if (classifier == "S" && score >= threshold) {
//        classification = "No-smile";
//    } else {
//        classification = "Smile";
//    }
//
//    classification += " : " + std::to_string(score);
//
//    // Display the result on the image
//    cv::putText(output, classification, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
//    cv::imshow("Result", output);
//    cv::waitKey(0);
//}
//
//int main(int argc, char** argv) {
//    std::string classifier = "M";  // Default is Mouth classifier
//    std::string image_path;
//    std::string CONFIG_FILE = false?"/Users/kapilsharma/GRA Work/facial-understanding/models":"/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/Final_code/models";
//    std::string model_path = CONFIG_FILE+"/face_detection_short.tflite";  // Specify the default model path
//    int landmarks_count = 40;      // Default number of landmarks
//    float threshold = 0.5;         // Default classification threshold
//
//    // Command-line argument parsing
//    for (int i = 1; i < argc; i++) {
//        if (std::string(argv[i]) == "--classifier") {
//            classifier = argv[++i];
//        } else if (std::string(argv[i]) == "--image") {
//            image_path = argv[++i];
//        } else if (std::string(argv[i]) == "--model") {
//            model_path = argv[++i];
//        } else if (std::string(argv[i]) == "--landmark") {
//            landmarks_count = std::stoi(argv[++i]);
//        } else if (std::string(argv[i]) == "--threshold") {
//            threshold = std::stof(argv[++i]);
//        }
//    }
//
//    if (image_path.empty()) {
//        std::cerr << "Please specify an image path using --image" << std::endl;
//        return -1;
//    }
//
//    auto interpreter = LoadModel(model_path);
//    if (!interpreter) {
//        std::cerr << "Failed to load model!" << std::endl;
//        return -1;
//    }
//
//    // Load the input image
//    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
//
//
//    if (image.empty()) {
//        std::cerr << "Failed to load image: " << image_path << std::endl;
//        return -1;
//    }
//
//    // Call run_single_image function
//    run_single_image(classifier, landmarks_count, image, interpreter, threshold);
//
//    return 0;
//}





















#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "FaceDetection_am6176.h"
#include "render.h"
#include "FaceLandmark.hpp"

// Forward declaration of cropFace
cv::Mat cropFace(const cv::Mat& image, const std::vector<Annotation>& renderData);


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

//    cv::imwrite("original_image.png", image);


    cv::Mat image_rgb;
    cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
    std::cout << "Converted to RGB Image Shape: " << image_rgb.rows << " x " << image_rgb.cols << " x " << image_rgb.channels() << std::endl;

//    cv::imwrite("BGRtoRGB_image.png", image_rgb);


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
    cv::Mat cropped_image;
    try {
        // Create FaceDetection instance using FRONT_CAMERA model
        FaceDetection face_detector(FaceDetection::FaceDetectionModel::FRONT_CAMERA);
        face_detector.printInterpreterInfo();

        // Run the face detection model
        std::vector<Detection> detections = face_detector(image_rgb);
        // Print the number of faces detected
//        std::cout << "Number of faces detected: " << detections.size() << std::endl;

        // Generate render data for bounding boxes
        std::vector<Annotation> render_data = detections_to_render_data(
                detections, Colors::GREEN, Colors::BLUE, 2, 4, true);

        // Debugging: Print render data
        std::cout << "Render Data:" << std::endl;
        for (const auto& annotation : render_data) {
            std::cout << "  Color: (" << annotation.color.r << ", "
                      << annotation.color.g << ", " << annotation.color.b << ")" << std::endl;
            std::cout << "  Thickness: " << annotation.thickness << std::endl;
            std::cout << "  Normalized Positions: " << (annotation.normalized_positions ? "True" : "False") << std::endl;

            for (const auto& point : annotation.points) {
                std::cout << "    Keypoint: (" << point.x << ", " << point.y << ")" << std::endl;
            }

            std::cout << "    Bounding Box: ("
                      << annotation.rect.left << ", " << annotation.rect.top << ", "
                      << annotation.rect.right << ", " << annotation.rect.bottom << ")"
                      << std::endl;
        }

        // Render annotations onto the image
        cv::Mat annotated_image = render_to_image(render_data, image_rgb);
        cv::cvtColor(annotated_image, annotated_image, cv::COLOR_RGB2BGR);
        cv::imshow("Annotated", annotated_image);



        cv::Mat cropped_face = cropFace(image, render_data);
        cv::imshow("Cropped Face", cropped_face);

        cv::Mat orig = cropped_face.clone(); // Equivalent to `cropped_face.copy()` in Python


        FaceLandmark face_landmark;  // Calls the constructor (equivalent to __init__)

//        std::vector<cv::Point2f> landmarks = face_landmark(cropped_face);  // Calls overloaded operator()



        int result = (classifier == "M") ? 1 : 0;  // Example: 1 for mouth, 0 for smile
        float score = 0.75f;  // Dummy confidence score

        return std::make_tuple(orig, cropped_face, result, score);

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

}



/* Crop Function with Black Background Padding */

//cv::Mat cropFace(const cv::Mat& image, const std::vector<Annotation>& renderData) {
//    if (renderData.empty()) {
//        throw std::runtime_error("No bounding box found in render data.");
//    }
//
//    // Use the first bounding box for cropping
//    const auto& rect = renderData[0].rect;
//
//    // Image dimensions
//    int imageWidth = image.cols;
//    int imageHeight = image.rows;
//
//    // Calculate actual bounding box pixel values
//    int actualLeft = static_cast<int>(rect.left * imageWidth);
//    int actualTop = static_cast<int>(rect.top * imageHeight);
//    int actualRight = static_cast<int>(rect.right * imageWidth);
//    int actualBottom = static_cast<int>(rect.bottom * imageHeight);
//
//    // Extend bounding box by 100 pixels
//    int x1 = actualLeft - 100;
//    int y1 = actualTop - 100;
//    int x2 = actualRight + 100;
//    int y2 = actualBottom + 100;
//
//    // Create a black background image of size (x2-x1, y2-y1)
//    cv::Mat paddedImage(cv::Size(x2 - x1, y2 - y1), image.type(), cv::Scalar(0, 0, 0));
//
//    // Calculate the region of intersection between the extended box and the original image
//    cv::Rect srcRegion(std::max(0, x1), std::max(0, y1),
//                       std::min(x2, imageWidth) - std::max(0, x1),
//                       std::min(y2, imageHeight) - std::max(0, y1));
//
//    // Calculate the destination region within the padded image
//    cv::Rect dstRegion(std::max(0, -x1), std::max(0, -y1), srcRegion.width, srcRegion.height);
//
//    // Copy the cropped region from the original image into the padded image
//    image(srcRegion).copyTo(paddedImage(dstRegion));
//
//    // Resize the padded image to 512x512
//    cv::resize(paddedImage, paddedImage, cv::Size(512, 512));
//
//    return paddedImage;
//}


cv::Mat cropFace(const cv::Mat& image, const std::vector<Annotation>& renderData) {
    if (renderData.empty()) {
        throw std::runtime_error("No bounding box found in render data.");
    }

    // Use the first bounding box for cropping
    const auto& rect = renderData[0].rect;

    // Make a copy of the original image
    cv::Mat orig = image.clone();
    int imageWidth = image.cols;
    int imageHeight = image.rows;

    // Calculating the actual pixel values of the bounding box
    int actualLeft = static_cast<int>(rect.left * imageWidth);
    int actualTop = static_cast<int>(rect.top * imageHeight);
    int actualRight = static_cast<int>(rect.right * imageWidth);
    int actualBottom = static_cast<int>(rect.bottom * imageHeight);

    // Extend the bounding box by 100 pixels on all sides
    int x1 = std::max(0, actualLeft - 100);
    int y1 = std::max(0, actualTop - 100);
    int x2 = std::min(imageWidth, actualRight + 100);
    int y2 = std::min(imageHeight, actualBottom + 100);

    // Crop the image
    cv::Rect cropRegion(x1, y1, x2 - x1, y2 - y1);
    cv::Mat cropped = orig(cropRegion);

    // Resize the cropped image to 512x512
    cv::resize(cropped, cropped, cv::Size(512, 512));

    return cropped;
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
    std::string CONFIG_FILE = false?"/Users/kapilsharma/GRA Work/facial-understanding/models":"/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/Final_code/models";
    std::string model_path = CONFIG_FILE+"/face_detection_short.tflite";  // Specify the default model path
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



