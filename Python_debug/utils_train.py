# -*- coding: utf-8 -*-
"""
utils_train.py
@description contains the crop functions used to train the smile and mouth classifier

@author: vivek chandra
"""
import os
import pandas as pd
from PIL import Image
from face import *
from land import *

# method to store the data in a csv file as image_path,cropped_image_path,all_landmarks,40_landmarks,label_mouth,label_smile
def crop_face(image,render_data):
  """
  This method is to crop the face with the rendered data available
  """
  orig = image.copy()
  image_width = image.size[0]
  image_height = image.size[1]

  # Normalized coordinates of the bounding box
  render_data
  left = render_data[0].data[0].left
  top = render_data[0].data[0].top
  right = render_data[0].data[0].right
  bottom = render_data[0].data[0].bottom

  # Calculating the actual pixel values of the bounding box
  actual_left = left * image_width
  actual_top = top * image_height
  actual_right = right * image_width
  actual_bottom = bottom * image_height

  cropped = orig.crop((actual_left-100, actual_top-100, actual_right+100, actual_bottom+100))
  cropped = cropped.resize((512,512))

  return cropped

def crop_mouth_region_from_tuples(image, landmarks):

    mouth_landmarks = landmarks  # Using all provided landmarks

    # Extract the minimum and maximum x and y coordinates from the mouth landmarks
    min_x = min(x for x, y in mouth_landmarks)
    max_x = max(x for x, y in mouth_landmarks)
    min_y = min(y for x, y in mouth_landmarks)
    max_y = max(y for x, y in mouth_landmarks)

    # Convert normalized coordinates to pixel coordinates
    img_width, img_height = image.size
    left = int(min_x * img_width)
    right = int(max_x * img_width)
    top = int(min_y * img_height)
    bottom = int(max_y * img_height)

    # Crop and return the mouth region
    return image.crop((left, top, right, bottom))

def crop_faces(src_path,destination_path_crop,cropped_mouth_path,label):
  """
  This method is to load -> face detect -> crop -> Face landmark detect -> full landmark -> 40 landmark -> save cropped
  This model
  """
  orig_path_list = []
  cropped_path_list = []
  full_lm = []
  mouth_lm = []
  for index,image_file in enumerate(os.listdir(src_path)):
    print(index)
    image = Image.open(os.path.join(src_path,image_file))
    face_detector = FaceDetection(model_type=FaceDetectionModel.BACK_CAMERA)
    face = face_detector(image)
    render_data = detections_to_render_data(face, bounds_color=Colors.GREEN)
    cropped_face = crop_face(image,render_data)
    orig = cropped_face.copy()
    cropped_path = os.path.join(destination_path_crop,image_file)
    cropped_face.save(cropped_path)
    # takes the face landmark
    face_landmark = FaceLandmark()
    landmarks = face_landmark(cropped_face)
    render_data,full_landmarks = face_landmarks_to_render_data(landmarks,Colors.RED,Colors.GREEN)
    mouth_landmarks = full_landmarks[:40]
    # crop the mouth region and save to a path
    cropped_image = crop_mouth_region_from_tuples(orig,mouth_landmarks)
    cropped_image.save(os.path.join(cropped_mouth_path,image_file))

    full_image_path = os.path.join(destination_path_crop,image_file)
    cropped_image_path = os.path.join(cropped_mouth_path,image_file)

    orig_path_list.append(full_image_path)
    cropped_path_list.append(cropped_image_path)
    full_lm.append(full_landmarks)
    mouth_lm.append(mouth_landmarks)










