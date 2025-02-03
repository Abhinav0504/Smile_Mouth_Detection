#target_sources(${APP_NAME}
#    PRIVATE
#        ${CMAKE_CURRENT_SOURCE_DIR}/run.cpp
#        ModelLoader.cpp
#        ModelLoader.hpp
#        DetectionPostProcess.cpp
#        DetectionPostProcess.hpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/IrisLandmark.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/IrisLandmark.hpp
#        FaceLandmark.cpp
#        FaceLandmark.hpp
#        FaceDetection.cpp
#        FaceDetection.hpp
#)

#
## Specify the minimum version of CMake
#cmake_minimum_required(VERSION 3.17)
#
## Define your project name
#project(FacialUnderstanding)
#
## Set the C++ standard
#set(CMAKE_CXX_STANDARD 17)
#
## Specify where TensorFlow Lite headers are located
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
#
## Link to the TensorFlow Lite library
#add_library(tensorflowlite SHARED IMPORTED)
#set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libs/mac/libtensorflowlite.dylib)
#
## Add your executable (the main file of your project)
#add_executable(${FacialUnderstanding} run.cpp)
#
## Link TensorFlow Lite to your executable
#target_link_libraries(${FacialUnderstanding} tensorflowlite)
#
## Add sources to your project
#target_sources(${FacialUnderstanding}
#        PRIVATE
#        ${CMAKE_CURRENT_SOURCE_DIR}/run.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/ModelLoader.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/ModelLoader.hpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/DetectionPostProcess.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/DetectionPostProcess.hpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/IrisLandmark.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/IrisLandmark.hpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/FaceLandmark.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/FaceLandmark.hpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/FaceDetection.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/FaceDetection.hpp
#)



# Link OpenCV libraries to your executable
#target_link_libraries(FacialUnderstanding ${OpenCV_LIBS})



# Specify the minimum version of CMake
cmake_minimum_required(VERSION 3.17)

# Define your project name
project(FacialUnderstanding)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add this at the top or after the project declaration in CMakeLists.txt
set(CMAKE_OSX_SYSROOT /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk)

# Add this line after setting CMAKE_OSX_SYSROOT
# Set the compiler flags explicitly using the fixed SDK path
set(CMAKE_CXX_FLAGS "-isysroot ${CMAKE_OSX_SYSROOT} -I/usr/include -I${CMAKE_OSX_SYSROOT}/usr/include")

# Find OpenCV package (add this right after defining the project)
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})
# Specify where TensorFlow Lite headers are located
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link to the TensorFlow Lite library
add_library(tensorflowlite SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libs/mac/libtensorflowlite.dylib)

# Add your executable (the main file of your project)
add_executable(FacialUnderstanding am6176_run.cpp)

# Link TensorFlow Lite and OpenCV to your executable
target_link_libraries(FacialUnderstanding tensorflowlite ${OpenCV_LIBS})

# Add sources to your project
target_sources(FacialUnderstanding
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/am6176_run.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/ModelLoader.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/ModelLoader.hpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/DetectionPostProcess.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/DetectionPostProcess.hpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/IrisLandmark.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/IrisLandmark.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/FaceLandmark.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/FaceLandmark.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/FaceDetection_am6176.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/FaceDetection_am6176.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Config.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Config.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Types.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Types.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Transforms.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Transforms.h
        ${CMAKE_CURRENT_SOURCE_DIR}/NMS.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/NMS.h
        ${CMAKE_CURRENT_SOURCE_DIR}/render.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render.h


)
