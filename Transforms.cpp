// transforms.cpp
#include "Transforms.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Types.h"


// Convert image to tensor data
ImageTensor image_to_tensor(const cv::Mat& image, Rect& roi, cv::Size output_size, bool keep_aspect_ratio, std::pair<float, float> output_range, bool flip_horizontal) {

    std::cout << std::endl << "STEP 4: Image_To_Tensor Function" << std::endl;
    std::cout << "---------------------------" << std::endl;

//    cv::imwrite("image.png", image);




    // Print the first 10 RGB pixels
    std::cout << "First 10 RGB pixels of the image:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        cv::Vec3b pixel = image.at<cv::Vec3b>(i / image.cols, i % image.cols);
        std::cout << "[" << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << "] " << std::endl;
    }
    std::cout << std::endl;

    // Print the last 10 RGB pixels
    std::cout << "Last 10 RGB pixels of the image:" << std::endl;
    for (int i = image.total() - 10; i < image.total(); ++i) {
        cv::Vec3b pixel = image.at<cv::Vec3b>(i / image.cols, i % image.cols);
        std::cout << "[" << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << "] " << std::endl;
    }
    std::cout << std::endl;


    // STEP 4.1: NORMALIZE IMAGE TO RGB
    cv::Mat normalized_image = _normalize_image(image);



    std::cout << std::endl << "     STEP 4.1: _normalize_to_image() results" << std::endl;
    std::cout << "      ---------------------------" << std::endl;
    std::cout << "      Number of Channels: " << normalized_image.channels() << std::endl;
    std::cout << "      Normalized Image Size: " << normalized_image.cols << "x" << normalized_image.rows << std::endl;

    std::cout << "First 10 Normalized pixels of the image:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        cv::Vec3b pixel = normalized_image.at<cv::Vec3b>(i / normalized_image.cols, i % normalized_image.cols);
        std::cout << "[" << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << "] " << std::endl;
    }
    std::cout << std::endl;

    // Print the last 10 RGB pixels
    std::cout << "Last 10 Normalized pixels of the image:" << std::endl;
    for (int i = normalized_image.total() - 10; i < normalized_image.total(); ++i) {
        cv::Vec3b pixel = normalized_image.at<cv::Vec3b>(i / normalized_image.cols, i % normalized_image.cols);
        std::cout << "[" << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << "] " << std::endl;
    }
    std::cout << std::endl;




//  STEP 4.2 : ROI SCALED SIZE

    cv::Size image_size = normalized_image.size();
    roi = roi.scaled({static_cast<float>(image_size.width), static_cast<float>(image_size.height)});
    std::cout << std::endl << "      STEP 4.2: ROI Scaled Size: " << roi.size()[0] << "x" << roi.size()[1] << std::endl;
    std::cout << "      ---------------------------" << std::endl;


    // Step 3: Handle output size or ROI dimensions
    if (output_size.width == 0 || output_size.height == 0) {
        output_size = cv::Size(static_cast<int>(roi.size()[0]), static_cast<int>(roi.size()[1]));
    }
    std::cout << "      Output Size: " << output_size.width << "x" << output_size.height << std::endl;

    int width, height;
    if (keep_aspect_ratio) {
        width = static_cast<int>(roi.size()[0]);
        height = static_cast<int>(roi.size()[1]);
    } else {
        width = output_size.width;
        height = output_size.height;
    }
    std::cout << "      Width: " << width << ", Height: " << height << std::endl;



// STEP 4.3: ROI POINTS

    std::vector<cv::Point2f> src_points = {
            cv::Point2f(roi.x_center - roi.width / 2, roi.y_center - roi.height / 2),  // top-left
            cv::Point2f(roi.x_center + roi.width / 2, roi.y_center - roi.height / 2),  // top-right
            cv::Point2f(roi.x_center + roi.width / 2, roi.y_center + roi.height / 2),  // bottom-right
            cv::Point2f(roi.x_center - roi.width / 2, roi.y_center + roi.height / 2)   // bottom-left
    };

    std::vector<cv::Point2f> dst_points = {
            cv::Point2f(0., 0.),             // top-left
            cv::Point2f(static_cast<float>(width), 0.),  // top-right
            cv::Point2f(static_cast<float>(width), static_cast<float>(height)), // bottom-right
            cv::Point2f(0., static_cast<float>(height))  // bottom-left
    };

    std::cout << std::endl << "      STEP 4.3: ROI Points: " << roi.size()[0] << "x" << roi.size()[1] << std::endl;
    std::cout << "      ---------------------------" << std::endl;
    std::cout << "      Src Points: ";
    for (const auto& point : src_points) {
        std::cout << "(" << point.x << "," << point.y << ") ";
    }
    std::cout << std::endl;

    std::cout << "      Dst Points: ";
    for (const auto& point : dst_points) {
        std::cout << "(" << point.x << "," << point.y << ") ";
    }
    std::cout << std::endl;


//  STEP 4.4: PERSPECTIVE TRANSFORM MATRIX
//    cv::imwrite("normalized_image.png", normalized_image);


    cv::Mat perspective_matrix = _perspective_transform_coeff(src_points, dst_points);
    std::cout << "\n    STEP 4.4: Perspective Transform Matrix" << std::endl;
    std::cout << "      ---------------------------" << std::endl;
    std::cout <<perspective_matrix << std::endl;

    cv::Mat roi_image;
    cv::warpPerspective(normalized_image, roi_image, perspective_matrix, cv::Size(width, height), cv::INTER_LINEAR);
    std::cout << "  roi_image size: " << roi_image.cols << "x" << roi_image.rows << std::endl;

//    cv::imwrite("roi_image.png", roi_image);

    cv::Mat roi_image_float;
    roi_image.convertTo(roi_image_float, CV_32F);

    std::cout << "\nFirst 10 values after perspective transformation (C++):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        cv::Vec3f pixel = roi_image_float.at<cv::Vec3f>(i / roi_image_float.cols, i % roi_image_float.cols);
        std::cout << "[" << pixel[0] << ", " << pixel[1] << ", " << pixel[2] << "]" << std::endl;
    }

    std::cout << "\nLast 10 values after perspective transformation (C++):" << std::endl;
    for (int i = roi_image_float.total() - 10; i < roi_image_float.total(); ++i) {
        cv::Vec3f pixel = roi_image_float.at<cv::Vec3f>(i / roi_image_float.cols, i % roi_image_float.cols);
        std::cout << "[" << pixel[0] << ", " << pixel[1] << ", " << pixel[2] << "]" << std::endl;
    }






    // STEP 4.5: PADDING
    std::cout << "\n   STEP 4.5: Padding (C++)" << std::endl;
    std::cout << "     ---------------------------" << std::endl;

    float pad_x = 0.0f, pad_y = 0.0f;
    float out_aspect = static_cast<float>(output_size.height) / output_size.width;
    float roi_aspect = static_cast<float>(roi.height) / roi.width;
    int new_width = static_cast<int>(roi.width);
    int new_height = static_cast<int>(roi.height);

// Letterboxing to maintain aspect ratio
    if (keep_aspect_ratio) {
        if (out_aspect > roi_aspect) {
            new_height = static_cast<int>(roi.width * out_aspect);
            pad_y = (1.0f - roi_aspect / out_aspect) / 2;
        } else {
            new_width = static_cast<int>(roi.height / out_aspect);
            pad_x = (1.0f - out_aspect / roi_aspect) / 2;
        }

        std::cout << "     out_aspect = " << out_aspect << ", roi_aspect = " << roi_aspect << std::endl;
        std::cout << "     new_width = " << new_width << ", new_height = " << new_height << std::endl;

        // Crop and map subregion
        int pad_h = static_cast<int>(pad_x * new_width);
        int pad_v = static_cast<int>(pad_y * new_height);

        cv::Rect crop_rect(-pad_h, -pad_v, new_width, new_height);
        crop_rect &= cv::Rect(0, 0, roi_image.cols, roi_image.rows); // Ensure within bounds

        cv::Mat cropped_image = roi_image(crop_rect);

        // Resize cropped image to target dimensions
        cv::Mat resized_image;
        cv::resize(cropped_image, resized_image, cv::Size(output_size.width, output_size.height), 0, 0, cv::INTER_LINEAR);

        roi_image = resized_image.clone(); // Update roi_image for further processing

        cv::Mat tmp_cropped_img_bgr;
        cv::Mat tmp_resized_image_bgr;
        cv::cvtColor(cropped_image, tmp_cropped_img_bgr, cv::COLOR_RGB2BGR);
        cv::imwrite("tmp_cropped_img_bgr.png", tmp_cropped_img_bgr);

        cv::cvtColor(roi_image, tmp_resized_image_bgr, cv::COLOR_RGB2BGR); //
        cv::imwrite("tmp_resized_image_bgr.png", tmp_resized_image_bgr);

        std::cout << "     Resized roi_image to: " << output_size.width << "x" << output_size.height << std::endl;
    }

    if (flip_horizontal) {
        cv::flip(roi_image, roi_image, 1);
        std::cout << "     Image flipped horizontally." << std::endl;
    }

    roi_image.convertTo(roi_image_float, CV_32F);

// Debugging: Print first and last 10 pixel values
    std::cout << "\nFirst 10 values after padding and resizing (C++):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        cv::Vec3f pixel = roi_image_float.at<cv::Vec3f>(i / roi_image_float.cols, i % roi_image_float.cols);
        std::cout << "[" << pixel[0] << ", " << pixel[1] << ", " << pixel[2] << "]" << std::endl;
    }

    std::cout << "\nLast 10 values after padding  and resizing(C++):" << std::endl;
    for (int i = roi_image_float.total() - 10; i < roi_image_float.total(); ++i) {
        cv::Vec3f pixel = roi_image_float.at<cv::Vec3f>(i / roi_image_float.cols, i % roi_image_float.cols);
        std::cout << "[" << pixel[0] << ", " << pixel[1] << ", " << pixel[2] << "]" << std::endl;
    }





//    STEP 4.6: VALUE RANGE TRANSFORM

    float min_val = output_range.first;
    float max_val = output_range.second;
    std::cout << "\n    STEP 4.6: Applying Value Range Transform" << std::endl;
    std::cout << "      ---------------------------" << std::endl;
    std::cout << "      Applying value range transform: min_val = " << min_val << ", max_val = " << max_val << std::endl;
    cv::Mat tensor_data;
    roi_image.convertTo(tensor_data, CV_32F, (max_val - min_val) / 255.0, min_val);
    std::cout << "      Image size after value range transform: " << tensor_data.cols << "x" << tensor_data.rows << std::endl;
    // Access and print all 3 channels (BGR) of the first pixel
    cv::Vec3f first_pixel = tensor_data.at<cv::Vec3f>(0, 0);  // Use Vec3f for a 3-channel float matrix

    std::cout << "      First pixel value after transform: ["
              << first_pixel[0] << ", "
              << first_pixel[1] << ", "
              << first_pixel[2] << "]"
              << std::endl;

// C++: Print first 10 transformed values
    std::cout << "\nFirst 10 Value Range Transformed values (C++):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        cv::Vec3f pixel = tensor_data.at<cv::Vec3f>(i / tensor_data.cols, i % tensor_data.cols);
        std::cout << "[" << pixel[0] << ", " << pixel[1] << ", " << pixel[2] << "]" << std::endl;
    }

// C++: Print last 10 transformed values
    std::cout << "\nLast 10 Value Range Transformed values (C++):" << std::endl;
    for (int i = tensor_data.total() - 10; i < tensor_data.total(); ++i) {
        cv::Vec3f pixel = tensor_data.at<cv::Vec3f>(i / tensor_data.cols, i % tensor_data.cols);
        std::cout << "[" << pixel[0] << ", " << pixel[1] << ", " << pixel[2] << "]" << std::endl;
    }

    std::array<float, 4> padding = {pad_x, pad_y, pad_x, pad_y};

    return ImageTensor(tensor_data, padding, image_size);

}







// Stubs for other functions
std::vector<float> sigmoid(const std::vector<float>& data) {
    // Placeholder implementation
    std::vector<float> result;
    return result;
}

// Adjust detections to remove letterboxing
std::vector<Detection> detection_letterbox_removal(
        const std::vector<Detection>& detections,
        const std::array<float, 4>& padding) {
    /*
    Args:
        detections: List of detection results with relative coordinates.
        padding: Array of floats representing padding values [left, top, right, bottom].

    Returns:
        Adjusted detections with bounding boxes and keypoints scaled to remove letterboxing.
    */
    std::cout << std::endl << "\n\tSTEP 7.3: Detection Letterbox Removal" << std::endl;
    std::cout << "\t---------------------------" << std::endl;

    // Extract padding values
    float left = padding[0];
    float top = padding[1];
    float right = padding[2];
    float bottom = padding[3];

    // Calculate scaling factors
    float h_scale = 1.0f - (left + right);
    float v_scale = 1.0f - (top + bottom);

    // Prepare the vector for adjusted detections
    std::vector<Detection> adjusted_detections;

    // Iterate through each detection and adjust its data
    for (const auto& detection : detections) {
        std::vector<cv::Point2f> adjusted_keypoints;

        std::cout << "\tOriginal Detection Keypoints: ";
        for (const auto& point : detection.keypoints) {
            std::cout << "(" << point.x << ", " << point.y << ") ";
        }
        std::cout << std::endl;

        // Adjust keypoints to remove letterboxing
        for (const auto& point : detection.keypoints) {
            float adjusted_x = (point.x - left) / h_scale;
            float adjusted_y = (point.y - top) / v_scale;
            adjusted_keypoints.emplace_back(adjusted_x, adjusted_y);
        }

        std::cout << "\tAdjusted Detection Keypoints: ";
        for (const auto& point : adjusted_keypoints) {
            std::cout << "(" << point.x << ", " << point.y << ") ";
        }
        std::cout << std::endl;

        // Create a new Detection object with adjusted keypoints and the same score
        Detection adjusted_detection(adjusted_keypoints, detection.score);
        adjusted_detections.push_back(adjusted_detection);
    }

    // Print out the adjusted detections for debugging purposes
    std::cout << "Adjusted Detections: " << adjusted_detections.size() << " detections adjusted." << std::endl;

    return adjusted_detections;
}


cv::Rect bbox_to_roi(const cv::Rect& bbox, const cv::Size& image_size) {
    // Placeholder implementation
    return bbox;
}

// Calculate the perspective transform matrix from source to destination points
cv::Mat _perspective_transform_coeff(const std::vector<cv::Point2f>& src_points, const std::vector<cv::Point2f>& dst_points) {
    // Get the 3x3 perspective transformation matrix using OpenCV
    cv::Mat perspective_matrix = cv::getPerspectiveTransform(src_points, dst_points);
    return perspective_matrix;
}



// Normalize image: Convert to RGB if necessary
cv::Mat _normalize_image(const cv::Mat& image) {
    cv::Mat normalized_image;
    // Check if the image is already in 3-channel RGB format
    if (image.channels() == 3) {
        normalized_image = image;
    } else if (image.channels() == 4) {
        // Convert from 4-channel (e.g., RGBA) to 3-channel (RGB)
        cv::cvtColor(image, normalized_image, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        // Convert from grayscale to RGB
        cv::cvtColor(image, normalized_image, cv::COLOR_GRAY2BGR);
    } else {
        throw std::runtime_error("Unsupported image format");
    }
    return normalized_image;
}


//cv::Rect convert_to_cv_rect(const Rect& roi, const cv::Size& image_size) {
//    int roi_x = static_cast<int>(roi.x_center * image_size.width - roi.width * image_size.width / 2);
//    int roi_y = static_cast<int>(roi.y_center * image_size.height - roi.height * image_size.height / 2);
//    int roi_w = static_cast<int>(roi.width * image_size.width);
//    int roi_h = static_cast<int>(roi.height * image_size.height);
//    return cv::Rect(roi_x, roi_y, roi_w, roi_h);
//}


