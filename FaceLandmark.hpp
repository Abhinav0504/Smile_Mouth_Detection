#ifndef FaceLandmark_HPP
#define FaceLandmark_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include "render.h"
#include "Types.h"

// Constants
const std::string MODEL_NAME = "face_landmark.tflite";
const int NUM_DIMS = 3;
const int NUM_LANDMARKS = 468;
const float ROI_SCALE[] = {1.5f, 1.5f};
const float DETECTION_THRESHOLD = 0.5f;


// Global landmark connections
const std::vector<std::pair<int, int>> FACE_LANDMARK_CONNECTIONS = {
        // Lips
        {61, 146}, {146, 91}, {91, 181}, {181, 84}, {84, 17}, {17, 314},
        {314, 405}, {405, 321}, {321, 375}, {375, 291}, {61, 185}, {185, 40},
        {40, 39}, {39, 37}, {37, 0}, {0, 267}, {267, 269}, {269, 270},
        {270, 409}, {409, 291}, {78, 95}, {95, 88}, {88, 178}, {178, 87},
        {87, 14}, {14, 317}, {317, 402}, {402, 318}, {318, 324}, {324, 308},
        {78, 191}, {191, 80}, {80, 81}, {81, 82}, {82, 13}, {13, 312},
        {312, 311}, {311, 310}, {310, 415}, {415, 308},
        // Left Eye
        {33, 7}, {7, 163}, {163, 144}, {144, 145}, {145, 153}, {153, 154},
        {154, 155}, {155, 133}, {33, 246}, {246, 161}, {161, 160}, {160, 159},
        {159, 158}, {158, 157}, {157, 173}, {173, 133},
        // Left Eyebrow
        {46, 53}, {53, 52},
        {52, 65}, {65, 55}, {70, 63}, {63, 105}, {105, 66}, {66, 107},
        // Right eye.
        {263, 249}, {249, 390}, {390, 373}, {373, 374}, {374, 380}, {380, 381},
        {381, 382}, {382, 362}, {263, 466}, {466, 388}, {388, 387}, {387, 386},
        {386, 385}, {385, 384}, {384, 398}, {398, 362},
        // Right eyebrow.
        {276, 283}, {283, 282},{282, 295}, {295, 285}, {300, 293}, {293, 334}, {334, 296}, {296, 336},
        // Face Oval.
        {10, 338}, {338, 297}, {297, 332}, {332, 284}, {284, 251}, {251, 389},
        {389, 356}, {356, 454}, {454, 323}, {323, 361}, {361, 288}, {288, 397},
        {397, 365}, {365, 379}, {379, 378}, {378, 400}, {400, 377}, {377, 152},
        {152, 148}, {148, 176}, {176, 149}, {149, 150}, {150, 136}, {136, 172},
        {172, 58}, {58, 132}, {132, 93}, {93, 234}, {234, 127}, {127, 162},
        {162, 21}, {21, 54}, {54, 103}, {103, 67}, {67, 109}, {109, 10}
};

class FaceLandmark {
private:
    std::unique_ptr<tflite::Interpreter> interpreter;
    int inputIndex;
    std::vector<int> inputShape;
    int dataIndex;
    int faceIndex;

public:
    // Constructor
    FaceLandmark(const std::string& modelPath = "");

    // Callable operator
    std::vector<cv::Point3f> operator()(const cv::Mat& image, const cv::Rect2f& roi = cv::Rect2f());
};

// Function prototypes
cv::Rect2f face_detection_to_roi(const Detection& face_detection, const cv::Size& image_size);
std::vector<cv::Point3f> project_landmarks(const cv::Mat& raw_data, const cv::Size& tensor_size,
                                           const cv::Size& image_size, const cv::Scalar& padding,
                                           const cv::Rect2f& roi);
std::vector<Annotation> face_landmarks_to_render_data(
        const std::vector<cv::Point3f>& face_landmarks,
        const Color& landmark_color,
        const Color& connection_color,
        float thickness = 2.0f,
        std::vector<Annotation>* output = nullptr);

#endif // FaceLandmark_HPP
