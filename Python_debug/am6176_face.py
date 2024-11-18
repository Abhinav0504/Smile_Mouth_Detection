# -*- coding: utf-8 -*-
"""
face.py
@description: to load the face model and return the faces detected

@author: vivek chandra
"""

from typing import List, Optional, Union
from typess import Rect,Detection
from am6176_transform import sigmoid, detection_letterbox_removal, image_to_tensor
from nms import non_maximum_suppression
import numpy as np
import os
import tensorflow as tf
from enum import IntEnum
from PIL import Image
from config import Config
from errors import InvalidEnumError
import cv2 as cv


MODEL_NAME_BACK = 'face_detection_back.tflite'
MODEL_NAME_FRONT = 'face_detection_front.tflite'
MODEL_NAME_SHORT = 'face_detection_short_range.tflite'
MODEL_NAME_FULL = 'face_detection_full_range.tflite'
MODEL_NAME_FULL_SPARSE = 'face_detection_full_range_sparse.tflite'

# score limit is 100 in mediapipe and leads to overflows with IEEE 754 floats
# this lower limit is safe for use with the sigmoid functions and float32
RAW_SCORE_LIMIT = 80
# threshold for confidence scores
MIN_SCORE = 0.5
# NMS similarity threshold
MIN_SUPPRESSION_THRESHOLD = 0.3

# from mediapipe module; irrelevant parts removed
# (reference: mediapipe/modules/face_detection/face_detection_front_cpu.pbtxt)
SSD_OPTIONS_FRONT = {
    'num_layers': 4,
    'input_size_height': 128,
    'input_size_width': 128,
    'anchor_offset_x': 0.5,
    'anchor_offset_y': 0.5,
    'strides': [8, 16, 16, 16],
    'interpolated_scale_aspect_ratio': 1.0
}

# (reference: modules/face_detection/face_detection_back_desktop_live.pbtxt)
SSD_OPTIONS_BACK = {
    'num_layers': 4,
    'input_size_height': 256,
    'input_size_width': 256,
    'anchor_offset_x': 0.5,
    'anchor_offset_y': 0.5,
    'strides': [16, 32, 32, 32],
    'interpolated_scale_aspect_ratio': 1.0
}

# (reference: modules/face_detection/face_detection_short_range_common.pbtxt)
SSD_OPTIONS_SHORT = {
    'num_layers': 4,
    'input_size_height': 128,
    'input_size_width': 128,
    'anchor_offset_x': 0.5,
    'anchor_offset_y': 0.5,
    'strides': [8, 16, 16, 16],
    'interpolated_scale_aspect_ratio': 1.0
}

# (reference: modules/face_detection/face_detection_full_range_common.pbtxt)
SSD_OPTIONS_FULL = {
    'num_layers': 1,
    'input_size_height': 192,
    'input_size_width': 192,
    'anchor_offset_x': 0.5,
    'anchor_offset_y': 0.5,
    'strides': [4],
    'interpolated_scale_aspect_ratio': 0.0
}


class FaceIndex(IntEnum):
    """Indexes of keypoints returned by the face detection model.

    Use these with detection results (by indexing the result):
    ```
        def get_left_eye_position(detection):
            x, y = detection[FaceIndex.LEFT_EYE]
            return x, y
    ```
    """
    LEFT_EYE = 0
    RIGHT_EYE = 1
    NOSE_TIP = 2
    MOUTH = 3
    LEFT_EYE_TRAGION = 4
    RIGHT_EYE_TRAGION = 5


class FaceDetectionModel(IntEnum):
    """Face detection model option:

    FRONT_CAMERA - 128x128 image, assumed to be mirrored

    BACK_CAMERA - 256x256 image, not mirrored

    SHORT - 128x128 image, assumed to be mirrored; best for short range images
            (i.e. faces within 2 metres from the camera)

    FULL - 192x192 image, assumed to be mirrored; dense; best for mid-ranges
           (i.e. faces within 5 metres from the camera)

    FULL_SPARSE - 192x192 image, assumed to be mirrored; sparse; best for
            mid-ranges (i.e. faces within 5 metres from the camera)
            this model is up ~30% faster than `FULL` when run on the CPU
    """
    FRONT_CAMERA = 0
    BACK_CAMERA = 1
    SHORT = 2
    FULL = 3
    FULL_SPARSE = 4


class FaceDetection1:
    """BlazeFace face detection model as used by Google MediaPipe.

    This model can detect multiple faces and returns a list of detections.
    Each detection contains the normalised [0,1] position and size of the
    detected face, as well as a number of keypoints (also normalised to
    [0,1]).

    The model is callable and accepts a PIL image instance, image file name,
    and Numpy array of shape (height, width, channels) as input. There is no
    size restriction, but smaller images are processed faster.

    Example:

    ```
        detect_faces = FaceDetection(model_path='/var/mediapipe/models')
        detections = detect_faces('/home/user/pictures/group_photo.jpg')
        print(f'num. faces found: {len(detections)}')
        # convert normalised coordinates to pixels (assuming 3kx2k image):
        if len(detections):
            rect = detections[0].bbox.scale(3000, 2000)
            print(f'first face rect.: {rect}')
        else:
            print('no faces found')
    ```

    Raises:
        InvalidEnumError: `model_type` contains an unsupported value
    """
    def __init__(
        self,
        model_type: FaceDetectionModel = FaceDetectionModel.FRONT_CAMERA,
        model_path: Optional[str] = None
    ) -> None:
        ssd_opts = {}
        model_p = Config.model_face
        if model_path is None:
            my_path = os.path.abspath(__file__)
            model_path = model_p
        if model_type == FaceDetectionModel.FRONT_CAMERA:
            self.model_path = model_p
            ssd_opts = SSD_OPTIONS_FRONT
        elif model_type == FaceDetectionModel.BACK_CAMERA:
            self.model_path = model_p
            ssd_opts = SSD_OPTIONS_BACK
        elif model_type == FaceDetectionModel.SHORT:
            self.model_path = model_p
            ssd_opts = SSD_OPTIONS_SHORT
        elif model_type == FaceDetectionModel.FULL:
            self.model_path = model_p
            ssd_opts = SSD_OPTIONS_FULL
        elif model_type == FaceDetectionModel.FULL_SPARSE:
            self.model_path = model_p
            ssd_opts = SSD_OPTIONS_FULL
        else:
            raise InvalidEnumError(f'unsupported model_type "{model_type}"')
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.input_shape = self.interpreter.get_input_details()[0]['shape']
        self.bbox_index = self.interpreter.get_output_details()[0]['index']
        self.score_index = self.interpreter.get_output_details()[1]['index']
        self.anchors = _ssd_generate_anchors(ssd_opts)

        # Print out model input/output details for debugging
        print("\nSTEP 2: FaceDetection Constructor Initialization")
        print("---------------------------")
        print(f"Model input index: {self.input_index}")
        print(f"Model input shape: {self.input_shape}")
        print(f"Bounding box output index: {self.bbox_index}")
        print(f"Score output index: {self.score_index}")

        # Generate SSD anchors and print for debugging
        self.anchors = _ssd_generate_anchors(ssd_opts)
        print(f"Generated {len(self.anchors)} anchors")

    def __call__(
        self,
        image: Union[Image, np.ndarray, str],
        roi: Optional[Rect] = None
    ) -> List[Detection]:
        """Run inference and return detections from a given image

        Args:
            image (Image|ndarray|str): Numpy array of shape
                `(height, width, 3)`, PIL Image instance or file name.

            roi (Rect|None): Optional region within the image that may
                contain faces.

        Returns:
            (list) List of detection results with relative coordinates.
        """
        print("\nSTEP 3: FaceDetection Object as a Function (__call__)")
        print("---------------------------")

        height, width = self.input_shape[1:3]

        print(f"Height : {height}")
        print(f"Width : {width}")

        # Check the type of 'image'
        if isinstance(image, np.ndarray):
            print(f"Image (numpy) shape: {image.shape}")
        elif isinstance(image, Image.Image):
            print(f"Image (PIL) size: {image.size}")
        elif isinstance(image, str):
            print(f"Image file path: {image}")

        print(f"ROI: {roi}")
        # Print ROI details
        print(f"ROI: {roi if roi else 'None provided (using default)'}")

        # Assuming transformations applied here, print image's first and last pixel values as additional checks
        if isinstance(image, Image.Image):
            image = np.array(image)  # Convert PIL.Image back to numpy array
            # cv.cvtColor(image, cv.COLOR_RGB2BGR)
            print(f"Converted to numpy array from PIL Image, shape: {image.shape}")
            print("First 10 pixels of image array:", image.reshape(-1, 3)[:10])
            print("Last 10 pixels of image array:", image.reshape(-1, 3)[-10:])


        image_data = image_to_tensor(
            image,
            roi,
            output_size=(width, height),
            keep_aspect_ratio=True,
            output_range=(-1, 1))



        # # Load the tensor data from the binary file
        # input_data = np.fromfile('/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/cpp_delivarables/build/input_data.bin', dtype=np.float32)
        # print("Loaded data size:", input_data.size)  # Should be 49152 for 128x128x3
        # # Check if the input data is grayscale (one channel)
        # if input_data.size == 128 * 128:  # Grayscale
        #     # Convert grayscale to RGB by duplicating the channel 3 times
        #     input_data = np.stack([input_data] * 3, axis=-1)
        #     print(f"Converted grayscale image to RGB with shape: {input_data.shape}")
        #
        # # Reshape as expected for RGB image
        # input_data = input_data.reshape((128, 128, 3))
        # print("Loaded tensor data shape:", input_data.shape)
        #
        #
        # # Assuming 'image_data.tensor_data' is a NumPy array
        # shape = image_data.tensor_data.shape
        # print(f"Image data tensor shape: {shape}")
        #
        # print("First few values of image data tensor:")
        # for i in range(10):  # Printing the first 10 values
        #     print(f"Python Image Data Tensor Value [{i}]: {image_data.tensor_data.flatten()[i]}")

        print("\nSTEP 5: Raw BBOX and Scores")
        print("---------------------------\n")

        input_data = image_data.tensor_data[np.newaxis]

        # input_data = np.expand_dims(input_data, axis=0) // INPUT FROM C++ CODE

        print(f"Input data shape: {input_data.shape}")

        # Print the entire array
        # print("Full input data tensor values:")
        # print(input_data)



        # print(f"Input tensor shape (with new axis): {input_data.shape}")
        # print(f"First pixel value in input tensor: {input_data[0, 0, 0, 0]}")  # Assuming 4D tensor [batch, height, width, channels]

        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()
        raw_boxes = self.interpreter.get_tensor(self.bbox_index)
        raw_scores = self.interpreter.get_tensor(self.score_index)
        print("\nShape of raw_boxes:", raw_boxes.shape)
        print("Shape of raw_scores:", raw_scores.shape)
        # print(f"Raw box data (first element): {raw_boxes.shape}")
        # print(f"Raw score data (first element): {raw_scores.shape}")

        # N = 3
        # M = 5
        # print(f"\nFirst {N} elements of raw_boxes:")
        # print(raw_boxes[0][:N])
        # print(f"\nFirst {M} elements of raw_scores:")
        # print(raw_scores[0][:M])

        print("\nFirst 10 raw_boxes:")
        for box in raw_boxes[0][:10]:
            print(box)

        print("\nLast 10 raw_boxes:")
        for box in raw_boxes[0][-10:]:
            print(box)

        # Print the first and last 10 raw scores
        print("\nFirst 10 raw_scores:")
        print(raw_scores[0][:10])

        print("\nLast 10 raw_scores:")
        print(raw_scores[0][-10:])


        print("\nSTEP 6: Decoded boxes and Sigmoid Scores")
        print("---------------------------\n")

        boxes = self._decode_boxes(raw_boxes)
        scores = self._get_sigmoid_scores(raw_scores)

        print("Length of boxes:", len(boxes))
        print("\nDecoded Boxes:")
        for i, box in enumerate(boxes[:5]):
            print(f"Box {i}: xmin={box[0][0]}, ymin={box[0][1]}, xmax={box[1][0]}, ymax={box[1][1]}")


        print("\nSigmoid Scores:")
        # for i, score in enumerate(scores[:10]):
        #     print(f"Score {i}: {score}")

        detections = FaceDetection1._convert_to_detections(boxes, scores)
        print(f"All face detections: {len(detections)}")

        pruned_detections = non_maximum_suppression(
                                detections,
                                MIN_SUPPRESSION_THRESHOLD, MIN_SCORE,
                                weighted=True)

        detections = detection_letterbox_removal(
            pruned_detections, image_data.padding)
        return detections

    def _decode_boxes(self, raw_boxes: np.ndarray) -> np.ndarray:
        """Decode raw boxes to (xmin, ymin, xmax, ymax) format."""
        scale = self.input_shape[1]
        num_points = raw_boxes.shape[-1] // 2

        # Step 1: Reshape and Scale
        boxes = raw_boxes.reshape(-1, num_points, 2) / scale
        print("\tStep 6.1 - Reshaped boxes shape:", boxes.shape)

        # Step 2: Anchor Adjustment
        boxes[:, 0] += self.anchors
        for i in range(2, num_points):
            boxes[:, i] += self.anchors

        # Step 3: Center Coordinates to Corner Coordinates
        center = np.array(boxes[:, 0])
        half_size = boxes[:, 1] / 2
        boxes[:, 0] = center - half_size  # xmin, ymin
        boxes[:, 1] = center + half_size  # xmax, ymax

        # Additional Debugging Print Statements
        print("\tAdjusted Anchors - Sample first box:", boxes[0])
        print("\tAdjusted Anchors - Full output sample:", boxes[:5])
        return boxes

    def _get_sigmoid_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """Extracted loop from ProcessCPU (line 327) in
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        """
        # just a single class ("face"), which simplifies this a lot
        # 1) thresholding; adjusted from 100 to 80, since sigmoid of [-]100
        #    causes overflow with IEEE single precision floats (max ~10e38)
        raw_scores[raw_scores < -RAW_SCORE_LIMIT] = -RAW_SCORE_LIMIT
        raw_scores[raw_scores > RAW_SCORE_LIMIT] = RAW_SCORE_LIMIT
        # 2) apply sigmoid function on clipped confidence scores
        return sigmoid(raw_scores)

    @staticmethod
    def _convert_to_detections(
        boxes: np.ndarray,
        scores: np.ndarray
    ) -> List[Detection]:
        """Apply detection threshold, filter invalid boxes and return
        detection instance.
        """
        # return whether width and height are positive
        def is_valid(box: np.ndarray) -> bool:
            return np.all(box[1] > box[0])

        score_above_threshold = scores > MIN_SCORE
        filtered_boxes = boxes[np.argwhere(score_above_threshold)[:, 1], :]
        filtered_scores = scores[score_above_threshold]
        return [Detection(box, score)
                for box, score in zip(filtered_boxes, filtered_scores)
                if is_valid(box)]


def _ssd_generate_anchors(opts: dict) -> np.ndarray:
    """This is a trimmed down version of the C++ code; all irrelevant parts
    have been removed.
    (reference: mediapipe/calculators/tflite/ssd_anchors_calculator.cc)
    """
    layer_id = 0
    num_layers = opts['num_layers']
    strides = opts['strides']
    assert len(strides) == num_layers
    input_height = opts['input_size_height']
    input_width = opts['input_size_width']
    anchor_offset_x = opts['anchor_offset_x']
    anchor_offset_y = opts['anchor_offset_y']
    interpolated_scale_aspect_ratio = opts['interpolated_scale_aspect_ratio']
    anchors = []
    while layer_id < num_layers:
        last_same_stride_layer = layer_id
        repeats = 0
        while (last_same_stride_layer < num_layers and
               strides[last_same_stride_layer] == strides[layer_id]):
            last_same_stride_layer += 1
            # aspect_ratios are added twice per iteration
            repeats += 2 if interpolated_scale_aspect_ratio == 1.0 else 1
        stride = strides[layer_id]
        feature_map_height = input_height // stride
        feature_map_width = input_width // stride
        for y in range(feature_map_height):
            y_center = (y + anchor_offset_y) / feature_map_height
            for x in range(feature_map_width):
                x_center = (x + anchor_offset_x) / feature_map_width
                for _ in range(repeats):
                    anchors.append((x_center, y_center))
        layer_id = last_same_stride_layer
    return np.array(anchors, dtype=np.float32)