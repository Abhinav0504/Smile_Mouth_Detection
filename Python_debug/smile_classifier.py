# -*- coding: utf-8 -*-
"""
smile_classifier.py
@description cnn+NN models with attention to train the smile classifier

@author: vivek chandra
"""

from face import FaceDetection, FaceDetectionModel
from land import FaceLandmark
from render import Colors, detections_to_render_data, render_to_image,landmarks_to_render_data
from land import *
from utils_train import *
from config import Config
from PIL import Image
import os
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import ast
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50,VGG16
from tensorflow.keras.layers import Input, Dense, Flatten, concatenate, Activation, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import ReduceLROnPlateau

def whole_face_40_lm_sf():
    # Load the VGG16 model, pretrained on ImageNet, without the top layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Image input branch
    image_input = base_model.input
    x1 = base_model.output
    x1 = Flatten()(x1)

    # Scale down the CNN features since they are less important
    cnn_weight = 0.7  # Adjust this weight to decrease the influence of CNN features
    # Expand dimension to match the x1 output shape
    cnn_weight_tensor = K.constant(cnn_weight)
    cnn_weight_tensor = K.reshape(cnn_weight_tensor, (1, 1))
    x1 = Multiply()([x1, cnn_weight_tensor])

    # Landmark input branch
    landmark_input = Input(shape=(40, 2))
    x2 = Flatten()(landmark_input)
    x2 = Dense(128, activation='relu')(x2)
    x2 = Dense(64, activation='relu')(x2)
    x2 = Dense(32, activation='relu')(x2)

    # Scale up the landmark features since they are more important
    landmark_weight = 1.0  # Adjust this weight to increase the influence of landmark features
    # Expand dimension to match the x2 output shape
    landmark_weight_tensor = K.constant(landmark_weight)
    landmark_weight_tensor = K.reshape(landmark_weight_tensor, (1, 1))
    x2 = Multiply()([x2, landmark_weight_tensor])

    # Merge branches
    combined = concatenate([x1, x2])

    # Classification layer
    classification = Dense(64, activation='relu')(combined)
    classification = Dense(2, activation='softmax')(classification)

    # Compile the model
    model = Model(inputs=[image_input, landmark_input], outputs=classification)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def whole_face_124_lm_sf():
    # Load the VGG16 model, pretrained on ImageNet, without the top layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Image input branch
    image_input = base_model.input
    x1 = base_model.output
    x1 = Flatten()(x1)

    # Scale down the CNN features since they are less important
    cnn_weight = 0.7  # Adjust this weight to decrease the influence of CNN features
    # Expand dimension to match the x1 output shape
    cnn_weight_tensor = K.constant(cnn_weight)
    cnn_weight_tensor = K.reshape(cnn_weight_tensor, (1, 1))
    x1 = Multiply()([x1, cnn_weight_tensor])

    # Landmark input branch
    landmark_input = Input(shape=(124, 2))
    x2 = Flatten()(landmark_input)
    x2 = Dense(128, activation='relu')(x2)
    x2 = Dense(64, activation='relu')(x2)
    x2 = Dense(32, activation='relu')(x2)

    # Scale up the landmark features since they are more important
    landmark_weight = 1.0  # Adjust this weight to increase the influence of landmark features
    # Expand dimension to match the x2 output shape
    landmark_weight_tensor = K.constant(landmark_weight)
    landmark_weight_tensor = K.reshape(landmark_weight_tensor, (1, 1))
    x2 = Multiply()([x2, landmark_weight_tensor])

    # Merge branches
    combined = concatenate([x1, x2])

    # Classification layer
    classification = Dense(64, activation='relu')(combined)
    classification = Dense(2, activation='softmax')(classification)

    # Compile the model
    model = Model(inputs=[image_input, landmark_input], outputs=classification)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import ReduceLROnPlateau
def whole_face_40_lm_attention():
    # Load the VGG16 model, pretrained on ImageNet, without the top layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Image input branch
    image_input = base_model.input
    x1 = base_model.output
    x1 = Flatten()(x1)

    # Landmark input branch
    landmark_input = Input(shape=(40, 2))
    x2 = Flatten()(landmark_input)
    x2 = Dense(128, activation='relu')(x2)
    x2 = Dense(64, activation='relu')(x2)
    x2 = Dense(32, activation='relu')(x2)

    # Merge branches
    combined = concatenate([x1, x2])
    importance_cnn = Dense(1, activation='sigmoid', use_bias=False, kernel_initializer=Constant(value=0.3))(x1)
    importance_landmarks = Dense(1, activation='sigmoid', use_bias=False, kernel_initializer=Constant(value=0.7))(x2)

    # Apply importance weights
    weighted_cnn = Multiply()([x1, importance_cnn])
    weighted_landmarks = Multiply()([x2, importance_landmarks])

    # Merge weighted features
    combined = concatenate([weighted_cnn, weighted_landmarks])

    # Classification layer
    classification = Dense(64, activation='relu')(combined)
    classification = Dense(2, activation='softmax')(classification)

    # Compile the model
    model = Model(inputs=[image_input, landmark_input], outputs=classification)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create the model

# Optional: model.summary() to see the model architecture

def whole_face_124_lm_attention():
    # Load the VGG16 model, pretrained on ImageNet, without the top layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Image input branch
    image_input = base_model.input
    x1 = base_model.output
    x1 = Flatten()(x1)

    # Landmark input branch
    landmark_input = Input(shape=(124, 2))
    x2 = Flatten()(landmark_input)
    x2 = Dense(128, activation='relu')(x2)
    x2 = Dense(64, activation='relu')(x2)
    x2 = Dense(32, activation='relu')(x2)

    combined = concatenate([x1, x2])
    importance_cnn = Dense(1, activation='sigmoid', use_bias=False, kernel_initializer=Constant(value=0.3))(x1)
    importance_landmarks = Dense(1, activation='sigmoid', use_bias=False, kernel_initializer=Constant(value=0.7))(x2)

    # Apply importance weights
    weighted_cnn = Multiply()([x1, importance_cnn])
    weighted_landmarks = Multiply()([x2, importance_landmarks])

    # Merge weighted features
    combined = concatenate([weighted_cnn, weighted_landmarks])

    # Classification layer
    classification = Dense(64, activation='relu')(combined)
    classification = Dense(2, activation='softmax')(classification)

    # Compile the model
    model = Model(inputs=[image_input, landmark_input], outputs=classification)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create the model

# Optional: model.summary() to see the model architecture

def load_and_prepare_data(csv_path):
    # Load the CSV file
    annotations = pd.read_csv(csv_path)

    # Preprocess the data
    annotations['Full_landmarks'] = annotations['Full_landmarks'].apply(ast.literal_eval)
    annotations['Mouth_landmarks'] = annotations['Mouth_landmarks'].apply(ast.literal_eval)
    annotations['Label'] = annotations['Label'].map({'smile': 1, 'No_smile': 0})  # Adjust mapping as necessary

    # Initialize lists to store our image data and landmarks
    image_data, mouth_data, landmark_data_124, landmark_data_40 = [], [], [], []

    for _, row in annotations.iterrows():
        # Load and preprocess the full image
        img = load_img(row['Image_path'], target_size=(224,224))
        img = img_to_array(img) / 255.0
        image_data.append(img)

        # Load and preprocess the cropped mouth image
        mouth = load_img(row['Cropped_path'], target_size=(64,64))
        mouth = img_to_array(mouth) / 255.0
        mouth_data.append(mouth)

        # Process landmarks
        landmarks_124 = np.array(row['Full_landmarks'])
        landmark_data_124.append(landmarks_124)
        landmarks_40 = np.array(row['Mouth_landmarks'])
        landmark_data_40.append(landmarks_40)

    # Convert lists to numpy arrays
    image_data = np.array(image_data)
    mouth_data = np.array(mouth_data)
    landmark_data_124 = np.array(landmark_data_124)
    landmark_data_40 = np.array(landmark_data_40)
    labels = to_categorical(annotations['Label'])  # Convert labels to one-hot encoding

    # Split the data into training and testing sets
    return train_test_split(image_data, mouth_data, landmark_data_124, landmark_data_40, labels, test_size=0.2, random_state=42)
csv_path = Config.smile_csv
image_train, image_test,mouth_train, mouth_test,landmark_124_train, landmark_124_test, landmark_40_train,landmark_40_test, y_train, y_test = load_and_prepare_data(smile_csv)

# training model 1:
model = whole_face_40_lm_sf()
checkpoint = ModelCheckpoint(
    '../models/best_model_cropped_smile_40_lm_sf_new.h5',          # Path where the model will be saved
    monitor='val_accuracy',   # Monitor validation accuracy for improvement
    verbose=1,                # Log when models are saved
    save_best_only=True,      # Only save a model if 'val_accuracy' has improved
    mode='max'                # 'max' means we look for the maximum validation accuracy
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,       # Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience=10,      # Number of epochs with no improvement after which learning rate will be reduced.
    min_lr=0.00001,   # Lower bound on the learning rate.
    verbose=1         # Int to print messages to stdout when reducing learning rate.
)
model.fit([mouth_train, landmark_40_train], y_train, validation_data=([mouth_test, landmark_40_test], y_test), epochs=50, batch_size=32,callbacks=[checkpoint,reduce_lr])

def load_and_prepare_data(csv_path):
    # Load the CSV file
    annotations = pd.read_csv(csv_path)

    # Preprocess the data
    annotations['Full_landmarks'] = annotations['Full_landmarks'].apply(ast.literal_eval)
    annotations['Mouth_landmarks'] = annotations['Mouth_landmarks'].apply(ast.literal_eval)
    annotations['Label'] = annotations['Label'].map({'smile': 1, 'No_smile': 0})  # Adjust mapping as necessary

    # Initialize lists to store our image data and landmarks
    image_data, mouth_data, landmark_data_124, landmark_data_40 = [], [], [], []

    for _, row in annotations.iterrows():
        # Load and preprocess the full image
        img = load_img(row['Image_path'], target_size=(224,224))
        img = img_to_array(img) / 255.0
        image_data.append(img)

        # Load and preprocess the cropped mouth image
        mouth = load_img(row['Cropped_path'], target_size=(64,64))
        mouth = img_to_array(mouth) / 255.0
        mouth_data.append(mouth)

        # Process landmarks
        landmarks_124 = np.array(row['Full_landmarks'])
        landmark_data_124.append(landmarks_124)
        landmarks_40 = np.array(row['Mouth_landmarks'])
        landmark_data_40.append(landmarks_40)

    # Convert lists to numpy arrays
    image_data = np.array(image_data)
    mouth_data = np.array(mouth_data)
    landmark_data_124 = np.array(landmark_data_124)
    landmark_data_40 = np.array(landmark_data_40)
    labels = to_categorical(annotations['Label'])  # Convert labels to one-hot encoding

    # Split the data into training and testing sets
    return train_test_split(image_data, mouth_data, landmark_data_124, landmark_data_40, labels, test_size=0.2, random_state=42)

model = whole_face_124_lm_sf()
checkpoint = ModelCheckpoint(
    '../models/best_model_cropped_smile_124_lm_sf_new.h5',          # Path where the model will be saved
    monitor='val_accuracy',   # Monitor validation accuracy for improvement
    verbose=1,                # Log when models are saved
    save_best_only=True,      # Only save a model if 'val_accuracy' has improved
    mode='max'                # 'max' means we look for the maximum validation accuracy
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,       # Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience=10,      # Number of epochs with no improvement after which learning rate will be reduced.
    min_lr=0.00001,   # Lower bound on the learning rate.
    verbose=1         # Int to print messages to stdout when reducing learning rate.
)
model.fit([mouth_train, landmark_124_train], y_train, validation_data=([mouth_test, landmark_124_test], y_test), epochs=50, batch_size=32,callbacks=[checkpoint,reduce_lr])

model = whole_face_40_lm_attention()
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,       # Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience=10,      # Number of epochs with no improvement after which learning rate will be reduced.
    min_lr=0.00001,   # Lower bound on the learning rate.
    verbose=1         # Int to print messages to stdout when reducing learning rate.
)
checkpoint = ModelCheckpoint(
    '../models/best_model_cropped_smile_40_lm_attention_new.h5',          # Path where the model will be saved
    monitor='val_accuracy',   # Monitor validation accuracy for improvement
    verbose=1,                # Log when models are saved
    save_best_only=True,      # Only save a model if 'val_accuracy' has improved
    mode='max'                # 'max' means we look for the maximum validation accuracy
)
model.fit([mouth_train, landmark_40_train], y_train, validation_data=([mouth_test, landmark_40_test], y_test), epochs=50, batch_size=32,callbacks=[checkpoint,reduce_lr])

model = whole_face_124_lm_attention()
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,       # Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience=5,      # Number of epochs with no improvement after which learning rate will be reduced.
    min_lr=0.00001,   # Lower bound on the learning rate.
    verbose=1         # Int to print messages to stdout when reducing learning rate.
)

checkpoint = ModelCheckpoint(
    '../models/best_model_cropped_smile_124_lm_attention_new.h5',          # Path where the model will be saved
    monitor='val_accuracy',   # Monitor validation accuracy for improvement
    verbose=1,                # Log when models are saved
    save_best_only=True,      # Only save a model if 'val_accuracy' has improved
    mode='max'                # 'max' means we look for the maximum validation accuracy
)
model.fit([mouth_train, landmark_124_train], y_train, validation_data=([mouth_test, landmark_124_test], y_test), epochs=50, batch_size=8,callbacks=[checkpoint,reduce_lr])

