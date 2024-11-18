# -*- coding: utf-8 -*-
"""
test.py
@description: loads the model,reads the image,
--args takes arguments on smile or mouth class
returns the face image,landmarks on face and the class in a cv window side by side

@author: vivek chandra
"""

import cv2 as cv
import argparse
from render import Colors, detections_to_render_data, render_to_image, landmarks_to_render_data
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from utils_train import *
from PIL import Image
from config import Config
from am6176_face import FaceDetection1


def predict(image, model, classifier, land):
    # loads the cv2 read image, BGR

    print("\nSTEP 1: Load in the Image")
    print("---------------------------")
    # converts cv2 image to RGB
    print("Original Image Shape:", image.shape)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    print("Converted to RGB Image Shape:", image_rgb.shape)

    # Print the first 10 and last 10 RGB pixel values
    print("\nFirst 10 RGB values after BGR to RGB conversion:")
    for i in range(10):
        print(image_rgb[0][i])

    print("\nLast 10 RGB values after BGR to RGB conversion:")
    for i in range(10):
        print(image_rgb[-1][-i - 1])

    # converts image to PIL.Image
    image = Image.fromarray(image_rgb)
    print("Converted to PIL Image:", image.size)

    # load the face detection model
    face_detector = FaceDetection1()
    print("Face Detection Model Loaded")

    # # load the facial landmarks model
    face = face_detector(image)
    print(f"Number of faces detected: {len(face)}")
    render_data = detections_to_render_data(face, bounds_color=Colors.GREEN)

    cropped_face = crop_face(image, render_data)
    orig = cropped_face.copy()
    face_landmark = FaceLandmark()
    landmarks = face_landmark(cropped_face)
    render_data_lm, full_landmarks = face_landmarks_to_render_data(landmarks, Colors.RED, Colors.GREEN)
    mouth_landmarks = full_landmarks[:40]

    # crop the mouth region and save to a path
    output = render_to_image(render_data_lm, cropped_face)
    cropped_image = crop_mouth_region_from_tuples(orig, mouth_landmarks)

    # smile_model_40
    img_array = img_to_array(cropped_face.resize((224, 224)))
    img = np.expand_dims(img_array, axis=0)  # Model expects a batch of images, so add batch dimension
    img /= 255.0

    cropped_image_test = img_to_array(cropped_image.resize((64, 64)))
    cropped_image_test = np.expand_dims(cropped_image_test,
                                        axis=0)  # Model expects a batch of images, so add batch dimension
    cropped_image_test /= 255.0

    landmarks1 = np.expand_dims(full_landmarks, axis=0)
    landmarks2 = np.expand_dims(mouth_landmarks, axis=0)

    if classifier.upper() == "M" and land == 40:
        # cropped image
        temp = model.predict([cropped_image_test, landmarks2])
        result = np.argmax(temp)
    elif classifier.upper() == "M" and land == 124:
        temp = model.predict([cropped_image_test, landmarks1])
        result = np.argmax(temp)
    elif classifier.upper() == "S" and land == 40:
        temp = model.predict([img, landmarks2])
        result = np.argmax(temp)
    else:
        temp = model.predict([img, landmarks1])
        result = np.argmax(temp)

    return output, cropped_image, result, temp[0][0]


# def run_live(classifier, land, model, t, camera: int = 0, resize: float = 1):
#     """
#     Runs the face detector and the face landmarks using webcam
#
#     Parameters
#     ----------
#     face : the FaceDetector model
#     land : the FacialLandmark model
#     camera : int, optional
#         to load the webcam. The default is 0.
#     resize : float, optional
#         DESCRIPTION. The default is 0.5.
#
#     Returns
#     -------
#     cv.window(face->face landmark->result on the window bar)
#
#     """
#     # make a note to flip the front selfie to get the correct orientation
#     cam = cv.VideoCapture(camera)
#     last_frame_time = time.time()
#     while True:
#         ret, frame = cam.read()
#
#         if not ret:
#             print("Failed to grab image from webcam")
#             break
#         frame = cv.resize(frame, dsize=None, fx=resize, fy=resize)
#         frame = cv.flip(frame, 1)
#         try:
#             op1, op2, res, score = predict(frame, model, classifier, land)
#             # calculate the FPS
#             current_time = time.time()
#             fps, last_frame_time = 1 / (current_time - last_frame_time), current_time
#             cv.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
#
#             if classifier == "M" and score >= t:  # res==0:
#                 r = "Mouth-close"
#             elif classifier == "M" and score < t:  # res == 1:
#                 r = "Mouth-open"
#             elif classifier == "S" and score >= t:  # res == 0:
#                 r = "No-smile"
#             else:
#                 r = "Smile"
#             r = r + str(score)
#             cv.imshow("(Press q to quit", frame)
#             result1 = np.array(op1)
#             result = cv.cvtColor(result1, cv.COLOR_BGR2RGB)
#
#             cv.putText(result, r, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
#             cv.imshow("Result", result)
#             if cv.waitKey(1) == ord("q"):
#                 break
#         except Exception as e:
#             print("No Face found!!")


def run_single_image(classifier, land, image, model, t):
    '''
    Run the model on a single image

    Returns
    -------
    result:String

    '''
    predict(image, model, classifier, land)
    op1, op2, res, score = predict(image, model, classifier, land)
    if classifier == "M" and score > t:  # res==0:
        r = "Mouth-close"
    elif classifier == "M" and score < t:  # res == 1:
        r = "Mouth-open"
    elif classifier == "S" and score >= t:  # res == 0:
        r = "No-smile"
    else:
        r = "Smile"
    r = r + " : " + str(score)
    # for mouth 0 -> 1: 0:mouth close -> 1:mouth open
    # for smile 0 -> 1: 0:No-smile -> 1:Smile
    result = np.array(op1)
    result = cv.cvtColor(result, cv.COLOR_BGR2RGB)
    cv.imshow(f"Result: {r} {res}", result)
    if cv.waitKey(0) == ord("q"):
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters for running the test file')

    # Add arguments
    parser.add_argument('-C', '--classifier', type=str, default="M", help='M for Mouth classifier, S for smile model')
    parser.add_argument('-I', '--image', type=str, default=None, help='Path to the image file')
    parser.add_argument('-L', '--landmark', type=int, default=40, help='number of landmarks to use')
    parser.add_argument('-T', '--threshold', type=float, default=0.5, help='number of landmarks to use')

    # Parse arguments
    args = parser.parse_args()

    if args.classifier.upper() == "M":
        # smile classifier
        if args.landmark == 40:
            # 40 lm model
            model = load_model(Config.mouth_40, compile=False)
            model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            model = load_model(Config.mouth_124, compile=False)
            model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        if args.landmark == 40:
            model = load_model(Config.smile_40, compile=False)
            model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            model = load_model(Config.smile_124, compile=False)
            model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    # Access arguments
    if args.image is None:
        # run_live(args.classifier, args.landmark, model, args.threshold)
        pass
    else:
        image = cv.imread(args.image)

        run_single_image(args.classifier, args.landmark, image, model, args.threshold)

    cv.destroyAllWindows()

