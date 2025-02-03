//#include "Config.h"
//
//// Define the model paths and other constants
//const std::string Config::model_face = "/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/Final_code/models/face_detection_short.tflite";
//const std::string Config::model_land = "/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/Final_code/models/TensorFlowFacialLandmarksV1.tflite";
//
//const std::string Config::smile_csv = "/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/Final_code/dataset/smile_train.csv";
//const std::string Config::mouth_csv = "/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/Final_code/dataset/mouth_train.csv";
//
//const std::string Config::smile_40 = "/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/Final_code/models/best_model_whole_face_40_lm_attention.h5";
//const std::string Config::smile_124 = "/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/Final_code/models/best_model_whole_face_124_lm_attention.h5";
//
//const std::string Config::mouth_40 = "/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/Final_code/models/best_model_cropped_mouth_40_lm_attention.h5";
//const std::string Config::mouth_124 = "/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/Final_code/models/best_model_cropped_mouth_124_lm_attention.h5";



#include "Config.h"

std::string CONFIG_FILE = false?"/Users/kapilsharma/GRA Work/facial-understanding/models":"/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/Final_code/models";


// Define the model paths and other constants
const std::string Config::model_face = CONFIG_FILE+"/face_detection_short.tflite";
const std::string Config::model_land = CONFIG_FILE+"/TensorFlowFacialLandmarksV1.tflite";

//const std::string Config::smile_csv = "/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/Final_code/dataset/smile_train.csv";
//const std::string Config::mouth_csv = "/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/Final_code/dataset/mouth_train.csv";
const std::string Config::smile_csv = CONFIG_FILE + "/dataset/smile_train.csv";
const std::string Config::mouth_csv = CONFIG_FILE + "/dataset/mouth_train.csv";


const std::string Config::smile_40 = CONFIG_FILE+"best_model_whole_face_40_lm_attention.h5";
const std::string Config::smile_124 = CONFIG_FILE+"/best_model_whole_face_124_lm_attention.h5";

const std::string Config::mouth_40 = CONFIG_FILE+"/best_model_cropped_mouth_40_lm_attention.h5";
const std::string Config::mouth_124 = CONFIG_FILE+"/best_model_cropped_mouth_124_lm_attention.h5";

