#include "FaceDetection_am6176.h"
#include <iostream>
#include <tensorflow/lite/model.h>
#include "Config.h"
#include <vector>
#include "Transforms.h"
#include "Types.h"
#include <iomanip>
#include "NMS.h"


#include <fstream>




// Declare a vector to hold the shape instead of a raw pointer
std::vector<int> input_shape;

// Constants for scoring and thresholds
const float MIN_SCORE = 0.3;           // Threshold for detection confidence scores
const float RAW_SCORE_LIMIT = 80.0f;    // Clipping limit for raw scores (avoid overflow in sigmoid)
float MIN_SUPPRESSION_THRESHOLD = 0.3f;  // Suppression threshold
bool weighted = true;


FaceDetection::FaceDetection(FaceDetectionModel model_type, const std::string& model_path) {
    SSDOptions ssd_opts;

    // Use the provided model path or a default one from the config
    this->model_path = model_path.empty() ? Config::model_face : model_path;

    // Select SSD options based on the model type
    switch (model_type) {
        case FaceDetectionModel::FRONT_CAMERA:
            ssd_opts = {4, 128, 128, 0.5, 0.5, {8, 16, 16, 16}, 1.0};
            break;
        case FaceDetectionModel::BACK_CAMERA:
            ssd_opts = {4, 256, 256, 0.5, 0.5, {16, 32, 32, 32}, 1.0};
            break;
        case FaceDetectionModel::SHORT:
            ssd_opts = {4, 128, 128, 0.5, 0.5, {8, 16, 16, 16}, 1.0};
            break;
        case FaceDetectionModel::FULL:
            ssd_opts = {1, 192, 192, 0.5, 0.5, {4}, 0.0};
            break;
        case FaceDetectionModel::FULL_SPARSE:
            ssd_opts = {1, 192, 192, 0.5, 0.5, {4}, 0.0};
            break;
        default:
            throw std::runtime_error("Unsupported model type");
    }

    // Load the TFLite model
    auto model = tflite::FlatBufferModel::BuildFromFile(this->model_path.c_str());
    if (!model) {
        throw std::runtime_error("Failed to load model: " + this->model_path);
    }

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        throw std::runtime_error("Failed to build interpreter.");
    }

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Failed to allocate tensors.");
    }

    // Get input/output indices and tensor details
    input_index = interpreter->inputs()[0];

    // Copy the shape data from the tensor to a vector
    TfLiteTensor* tensor = interpreter->tensor(input_index);
    input_shape.assign(tensor->dims->data, tensor->dims->data + tensor->dims->size);  // Copy dimensions

//    bbox_index = interpreter->outputs()[0];
//    score_index = interpreter->outputs()[1];

    bbox_index = 175;
    score_index = 174;

    // Generate SSD anchors
    anchors = generate_anchors(ssd_opts);

    std::cout << "FaceDetection initialized successfully with model: " << this->model_path << std::endl;
}


// Print interpreter details for debugging
void FaceDetection::printInterpreterInfo() const {
    std::cout << std::endl << "STEP 2: FaceDetection Constructor Initialization" << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "Input index: " << input_index << std::endl;
    std::cout << "Input shape: [";
    for (size_t i = 0; i < input_shape.size(); ++i) {
        std::cout << input_shape[i];
        if (i < input_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Bounding box index: " << bbox_index << std::endl;
    std::cout << "Score index: " << score_index << std::endl;
    std::cout << "Number of anchors: " << anchors.size() << std::endl;

    std::cout << "First 10 anchors:" << std::endl;
    for (size_t i = 0; i < std::min(anchors.size(), static_cast<size_t>(10)); ++i) {
        std::cout << "[" << anchors[i].first << ", " << anchors[i].second << "]" << std::endl;
    }

    std::cout << "Last 10 anchors:" << std::endl;
    for (size_t i = (anchors.size() > 10 ? anchors.size() - 10 : 0); i < anchors.size(); ++i) {
        std::cout << "[" << anchors[i].first << ", " << anchors[i].second << "]" << std::endl;
    }

}


// Assuming 'input_data' is the C++ tensor data
void save_tensor_to_file(const float* data, int size, const std::string& file_path) {
    std::ofstream out_file(file_path, std::ios::binary);
    if (!out_file) {
        std::cerr << "Error: Could not open file for writing: " << file_path << std::endl;
        return;
    }
//    out_file.write(reinterpret_cast<const char*>(data), size * sizeof(float));
    out_file.write(reinterpret_cast<const char*>(data), 128 * 128 * 3 * sizeof(float));  // 128 * 128 * 3 for RGB

    out_file.close();
    std::cout << "Tensor data saved to file: " << file_path << std::endl;
}


std::vector<Detection> FaceDetection::operator()(const cv::Mat& image, Rect* roi) {
    std::cout << std::endl << "STEP 3: FaceDetection Object as a Function (Operator)" << std::endl;
    std::cout << "---------------------------" << std::endl;

    // Set height and width for the input tensor from the model input shape
    int height = input_shape[1];
    int width = input_shape[2];
    std::cout << "Height " << height << std::endl;
    std::cout << "Width " << width << std::endl;
    std::cout << "Image size: " << image.rows << " x " << image.cols << " x " << image.channels() << std::endl;

    // Check if roi is provided or assign default Rect for full image
    Rect default_roi(0.5f, 0.5f, 1.0f, 1.0f, 0.0f, true);  // Default ROI as in Python

    const Rect& actual_roi = roi ? *roi : default_roi;
    std::cout << "ROI: "
              << (roi ? "Custom ROI provided" : "Default ROI (full image)")
              << " [x_center=" << actual_roi.x_center
              << ", y_center=" << actual_roi.y_center
              << ", width=" << actual_roi.width
              << ", height=" << actual_roi.height
              << ", rotation=" << actual_roi.rotation
              << ", normalized=" << (actual_roi.normalized ? "True" : "False")
              << "]" << std::endl;

    // Print the first and last 10 pixels in the image array as an additional check
    std::cout << "First 10 RGB pixels of image array:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        cv::Vec3b pixel = image.at<cv::Vec3b>(i / image.cols, i % image.cols);
        std::cout << "[" << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << "] " << std::endl;
    }
    std::cout << "\nLast 10 RGB pixels of image array:" << std::endl;
    for (int i = image.total() - 10; i < image.total(); ++i) {
        cv::Vec3b pixel = image.at<cv::Vec3b>(i / image.cols, i % image.cols);
        std::cout << "[" << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << "] " << std::endl;
    }
    std::cout << std::endl;

    // Convert the image to tensor data using image_to_tensor
    ImageTensor image_data = image_to_tensor(image, roi ? *roi : default_roi, cv::Size(width, height), true, {-1.0f, 1.0f}, false);


    // Assuming 'image_data.tensor_data' is of type cv::Mat
    int height1 = image_data.tensor_data.rows;
    int width1 = image_data.tensor_data.cols;
    int channels1 = image_data.tensor_data.channels();

    std::cout << "Image data tensor shape: (" << height1 << ", " << width1 << ", " << channels1 << ")" << std::endl;

//    std::cout << "First few values of image data tensor:" << std::endl;
//    for (int i = 0; i < 10; ++i) {  // Printing the first 10 values
//        const float* ptr = image_data.tensor_data.ptr<float>(); // Assuming it's float
//        std::cout << "C++ Image Data Tensor Value [" << i << "]: " << ptr[i] << std::endl;
//    }


    // Example usage: Save 'input_data' tensor
    int input_data_size = image_data.tensor_data.total();
    save_tensor_to_file(image_data.tensor_data.ptr<float>(), input_data_size, "input_data.bin");






    std::cout << std::endl << "STEP 5: Raw BBOX and Scores" << std::endl;
    std::cout << "---------------------------" << std::endl;

    // Step 1: Get a pointer to the input tensor's data
    float* input_data = interpreter->typed_input_tensor<float>(input_index);

    int input_size = 1 * 128 * 128 * 3;


// Step 2: Copy the image tensor data into the input tensor
    std::memcpy(input_data, image_data.tensor_data.ptr<float>(), image_data.tensor_data.total() * sizeof(float));
//    std::memcpy(input_data, image_data.tensor_data.ptr<float>(), input_size * sizeof(float));

//    std::cout << "Full input data tensor values:" << std::endl;
//    for (int i = 0; i < input_size; ++i) {
//        std::cout << "Input Data Value [" << i << "]: " << input_data[i] << std::endl;
//    }


    int height2 = image_data.tensor_data.rows;
    int width2 = image_data.tensor_data.cols;
    int channels2 = image_data.tensor_data.channels();

// Print input tensor data before invoking the interpreter
    std::cout << "Input data shape: " << width2 << "x" << height2 << "x" << channels2 << std::endl;

    // Print the full input tensor shape
    TfLiteTensor* input_tensor = interpreter->tensor(input_index);
    std::cout << "Input data shape (all dimensions): [";
    for (int i = 0; i < input_tensor->dims->size; i++) {
        std::cout << input_tensor->dims->data[i];
        if (i < input_tensor->dims->size - 1) std::cout << "x";
    }
    std::cout << "]" << std::endl;



    std::cout << "First few values of the input data tensor:" << std::endl;


    // Debug: Print some info about the input tensor
    std::cout << "Input tensor size: " << image_data.tensor_data.size() << std::endl;
    std::cout << "First pixel value in input tensor: " << input_data[0] << std::endl;

    // Step 3: Invoke the interpreter
    TfLiteStatus status = interpreter->Invoke();
    if (status != kTfLiteOk) {
        std::cerr << "Error: Failed to invoke the interpreter." << std::endl;
        return {};
    }


    TfLiteTensor* bbox_tensor = interpreter->tensor(175);
    TfLiteTensor* score_tensor = interpreter->tensor(174);

    std::cout << std::endl << "Number of output tensors: " << interpreter->outputs().size() << std::endl;

//    std::cout << "Raw box tensor shape: [" << bbox_tensor->dims->data[0] << ", "
//              << bbox_tensor->dims->data[1] << ", " << bbox_tensor->dims->data[2] << "]" << std::endl;
//
//    std::cout << "Raw score tensor shape: [" << score_tensor->dims->data[0] << ", "
//              << score_tensor->dims->data[1] << ", " << score_tensor->dims->data[2] << "]" << std::endl;


// Additional debugging: print the shape of tensors
    std::cout << "Raw box tensor shape: [";
    for (int i = 0; i < bbox_tensor->dims->size; i++) {
        std::cout << bbox_tensor->dims->data[i];
        if (i < bbox_tensor->dims->size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Raw score tensor shape: [";
    for (int i = 0; i < score_tensor->dims->size; i++) {
        std::cout << score_tensor->dims->data[i];
        if (i < score_tensor->dims->size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;


    // Step 4: Get the output tensors (raw bounding boxes and scores)
//    float* raw_boxes = interpreter->typed_output_tensor<float>(bbox_index);
//    float* raw_scores = interpreter->typed_output_tensor<float>(score_index);
    float* raw_boxes = interpreter->typed_output_tensor<float>(0);
    float* raw_scores = interpreter->typed_output_tensor<float>(1);
//    std::cout << "Raw box data (first element): " << raw_boxes[0] << std::endl;
//    std::cout << "Raw score data (first element): " << raw_scores[0] << std::endl;

// Calculate the total number of boxes and scores based on tensor dimensions
    int total_raw_boxes = bbox_tensor->dims->data[1];  // Assuming [1, 896, 16]
    int total_raw_scores = score_tensor->dims->data[1]; // Assuming [1, 896, 1]

    std::cout << "Total number of raw boxes: " << total_raw_boxes << std::endl;
    std::cout << "Total number of raw scores: " << total_raw_scores << std::endl;

    // Check if these point to different memory locations
    if (raw_boxes == raw_scores) {
        std::cerr << "Error: raw_boxes and raw_scores point to the same memory!" << std::endl;
    } else {
        std::cout << "raw_boxes and raw_scores are correctly pointing to different data." << std::endl;
    }


    // Check if these point to different memory locations
    if (raw_boxes == raw_scores) {
        std::cerr << "Error: raw_boxes and raw_scores point to the same memory!" << std::endl;
        // Create a copy of raw_scores to avoid memory overlap
        float* raw_scores_copy = new float[score_tensor->bytes / sizeof(float)];
        std::memcpy(raw_scores_copy, raw_scores, score_tensor->bytes);
        raw_scores = raw_scores_copy;  // Use the copy of raw_scores from now on
    } else {
        std::cout << "raw_boxes and raw_scores are correctly pointing to different data." << std::endl;
    }


// Print the first and last 10 raw boxes (each with 16 elements)
    std::cout << "\nFirst 10 raw_boxes:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "[ ";
        for (int j = 0; j < 16; j++) {
            std::cout << raw_boxes[i * 16 + j] << " ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "\nLast 10 raw_boxes:" << std::endl;
    for (int i = total_raw_boxes - 10; i < total_raw_boxes; i++) {
        std::cout << "[ ";
        for (int j = 0; j < 16; j++) {
            std::cout << raw_boxes[i * 16 + j] << " ";
        }
        std::cout << "]" << std::endl;
    }

// Print the first and last 10 raw scores
    std::cout << "\nFirst 10 raw_scores:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << raw_scores[i] << std::endl;
    }

    std::cout << "\nLast 10 raw_scores:" << std::endl;
    for (int i = total_raw_scores - 10; i < total_raw_scores; i++) {
        std::cout << raw_scores[i] << std::endl;
    }





// STEP 6: Decoded Boxes and Sigmoid Scores
    std::cout << std::endl << "STEP 6: Decoded Boxes and Sigmoid Scores" << std::endl;
    std::cout << "---------------------------" << std::endl;


    // Perform box decoding and sigmoid scoring
    std::vector<std::vector<std::pair<float, float>>> boxes = decode_boxes(raw_boxes);

//    // Debug: Print final adjusted boxes before returning
//    std::cout << std::endl << "Final Adjusted Boxes (first 5 rows):" << std::endl;
//    for (int i = 0; i < std::min(5, 896); ++i) {
//        std::cout << "  Box " << i << ": [";
//        for (int j = 0; j < 8; ++j) {
//            std::cout << "[" << boxes[i][j].first << ", " << boxes[i][j].second << "]";
//            if (j < 8 - 1) std::cout << ", " << std::endl;
//        }
//        std::cout << "]" << std::endl;
//    }


    std::vector<float> scores = get_sigmoid_scores(raw_scores);

    // Print the shape of the scores (if applicable)
    std::cout << std::endl << "Sigmoid Scores Shape: (" << scores.size() << ")" << std::endl;

// Print the first 5 scores
    std::cout << "Sigmoid Scores (First 10): ";
    for (size_t i = 0; i < std::min(static_cast<size_t>(10), scores.size()); ++i) {
        std::cout << scores[i] << " ";
    }
    std::cout << std::endl;

// Print the last 5 scores
    std::cout << "Sigmoid Scores (Last 10): ";
    for (size_t i = scores.size() > 10 ? scores.size() - 10 : 0; i < scores.size(); ++i) {
        std::cout << scores[i] << " ";
    }
    std::cout << std::endl;

    // Pass the decoded boxes and scores directly to convert_to_detections
    std::vector<Detection> detections = convert_to_detections(boxes, scores);

    // Debug: Print the first 5 detections
    for (size_t i = 0; i < std::max(detections.size(), size_t(5)); ++i) {
        detections[i].print();
    }

    std::cout << std::endl << "STEP 7: Non-Maximum Suppression" << std::endl;
    std::cout << "---------------------------" << std::endl;

    std::vector<Detection> pruned_detections = non_maximum_suppression(
            detections,
            MIN_SUPPRESSION_THRESHOLD,
            MIN_SCORE,
            weighted
    );

    // Print pruned detections for debugging
    std::cout << std::endl << "Pruned Detections (" << pruned_detections.size() << "):" << std::endl;
    for (size_t i = 0; i < pruned_detections.size(); ++i) {
        std::cout << "Detection " << i << ": ";
        pruned_detections[i].print();
    }

    std::cout << "\nRemoving letterboxing from detections..." << std::endl;

// Remove letterboxing adjustments
    std::vector<Detection> adjusted_detections = detection_letterbox_removal(pruned_detections, image_data.padding);
    std::cout << "Type of image_data.padding: " << typeid(image_data.padding).name() << std::endl;
    std::cout << "Contents of image_data.padding: [";
    for (size_t i = 0; i < image_data.padding.size(); ++i) {
        std::cout << image_data.padding[i];
        if (i < image_data.padding.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;



    return adjusted_detections;
}




std::vector<std::vector<std::pair<float, float>>> FaceDetection::decode_boxes(const float* raw_boxes) {
    int num_points = 8;  // Matches the Python `reshape(-1, num_points, 2)`
    int num_boxes = 896; // Matches Python anchors
    float scale = static_cast<float>(input_shape[1]); // Input height for scaling
    std::cout << "  Scale: " << scale << std::endl;

    // Step 1: Reshape and Scale
    std::vector<std::vector<std::pair<float, float>>> boxes(num_boxes, std::vector<std::pair<float, float>>(num_points));
    for (int i = 0; i < num_boxes; ++i) {
        for (int j = 0; j < num_points; ++j) {
            float x = raw_boxes[i * num_points * 2 + j * 2] / scale;
            float y = raw_boxes[i * num_points * 2 + j * 2 + 1] / scale;
            boxes[i][j] = {x, y};
        }
    }

    // Debug: Print first 5 rows of scaled boxes
    std::cout << "Scaled Boxes (first 5 rows):" << std::endl;
    for (int i = 0; i < std::min(5, num_boxes); ++i) {
        std::cout << "  Box " << i << ": [";
        for (int j = 0; j < num_points; ++j) {
            std::cout << "[" << boxes[i][j].first << ", " << boxes[i][j].second << "]";
            if (j < num_points - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Step 2: Anchor Adjustment
    for (int i = 0; i < num_boxes; ++i) {
        boxes[i][0].first += anchors[i].first;
        boxes[i][0].second += anchors[i].second;

        for (int j = 2; j < num_points; ++j) {
            boxes[i][j].first += anchors[i].first;
            boxes[i][j].second += anchors[i].second;
        }
    }

    // Debug: Print adjusted anchors
    std::cout << std::endl << "Adjusted Anchors (first 5 boxes):" << std::endl;
    for (int i = 0; i < std::min(5, num_boxes); ++i) {
        std::cout << "  Box " << i << ": [";
        for (int j = 0; j < num_points; ++j) {
            std::cout << "[" << boxes[i][j].first << ", " << boxes[i][j].second << "]";
            if (j < num_points - 1) std::cout << ", " << std::endl;
        }
        std::cout << "]" << std::endl;
    }

    // Step 3: Modify first two points in each box
    for (int i = 0; i < num_boxes; ++i) {
        float x_center = boxes[i][0].first;
        float y_center = boxes[i][0].second;
        float half_width = boxes[i][1].first / 2.0f;
        float half_height = boxes[i][1].second / 2.0f;

        boxes[i][0] = {x_center - half_width, y_center - half_height};  // [xmin, ymin]
        boxes[i][1] = {x_center + half_width, y_center + half_height};  // [xmax, ymax]
    }

    // Debug: Print final adjusted boxes before returning
    std::cout << std::endl << "Final Adjusted Boxes (first 5 rows):" << std::endl;
    for (int i = 0; i < std::min(5, num_boxes); ++i) {
        std::cout << "  Box " << i << ": [";
        for (int j = 0; j < num_points; ++j) {
            std::cout << "[" << boxes[i][j].first << ", " << boxes[i][j].second << "]";
            if (j < num_points - 1) std::cout << ", " << std::endl;
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "\nTotal number of boxes being returned: " << boxes.size() << std::endl;


    return boxes;
}







// Apply sigmoid function to raw scores
std::vector<float> FaceDetection::get_sigmoid_scores(const float* raw_scores) {
    std::vector<float> scores;
    int num_scores = 896;  // Assuming same size as number of boxes

    for (int i = 0; i < num_scores; ++i) {
        float clipped_score = std::max(std::min(raw_scores[i], 80.0f), -80.0f);  // Clipping between -80 and 80
        scores.push_back(1.0f / (1.0f + std::exp(-clipped_score)));  // Apply sigmoid
    }
    std::cout << "\nTotal number of Scores being returned: " << scores.size() << std::endl;

    return scores;
}



std::vector<Detection> FaceDetection::convert_to_detections(
        const std::vector<std::vector<std::pair<float, float>>>& boxes,
        const std::vector<float>& scores
) {
    std::cout << "\n\tStep 6.2 - Convert To Detections:" << std::endl;
    std::cout << "\t---------------------------------" << std::endl;

    // Lambda to check if a box is valid
    auto is_valid = [](const std::vector<std::pair<float, float>>& box) -> bool {
        return box[1].first > box[0].first && box[1].second > box[0].second;
    };

    // Filter scores above threshold
    std::vector<size_t> score_above_threshold_indices;
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i] > MIN_SCORE) {
            score_above_threshold_indices.push_back(i);
        }
    }

    std::cout << "\tFiltered scores count: " << score_above_threshold_indices.size() << std::endl;

    // Filter boxes and scores
    std::vector<Detection> detections;
    for (size_t idx : score_above_threshold_indices) {
        const auto& box = boxes[idx];
        std::vector<cv::Point2f> keypoints;

        // Convert [xmin, ymin, xmax, ymax, ...] pairs into cv::Point2f
        for (const auto& point : box) {
            keypoints.emplace_back(point.first, point.second);
        }

        // Create detection if valid
        if (is_valid(box)) {
            detections.emplace_back(keypoints, scores[idx]);
        }
    }

    std::cout << "\tTotal detections: " << detections.size() << std::endl;

    return detections;
}








// Generate SSD anchors (equivalent to _ssd_generate_anchors in Python)
std::vector<std::pair<float, float>> FaceDetection::generate_anchors(const SSDOptions& options) {
    std::vector<std::pair<float, float>> anchors;
    int layer_id = 0;

    while (layer_id < options.num_layers) {
        int last_same_stride_layer = layer_id;
        int repeats = 0;
        while (last_same_stride_layer < options.num_layers &&
               options.strides[last_same_stride_layer] == options.strides[layer_id]) {
            last_same_stride_layer++;
            repeats += (options.interpolated_scale_aspect_ratio == 1.0) ? 2 : 1;
        }

        int stride = options.strides[layer_id];
        int feature_map_height = options.input_size_height / stride;
        int feature_map_width = options.input_size_width / stride;

        for (int y = 0; y < feature_map_height; ++y) {
            float y_center = (y + options.anchor_offset_y) / feature_map_height;
            for (int x = 0; x < feature_map_width; ++x) {
                float x_center = (x + options.anchor_offset_x) / feature_map_width;
                for (int i = 0; i < repeats; ++i) {
                    anchors.emplace_back(x_center, y_center);
                }
            }
        }

        layer_id = last_same_stride_layer;
    }

    return anchors;
}