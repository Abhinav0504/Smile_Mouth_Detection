import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# Replace with your model's file path
MODEL_PATH = r"/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/Final_code/models/face_detection_short.tflite"

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensor indices
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define input size
input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1:3]

# Load and preprocess an image
image_path = r"/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/facial-understanding-Smile_Mouth_TF/dataset/smile(0)-mouth(1)/00096_0.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (input_width, input_height))

# Normalize the image as per the model's requirements
input_data = np.expand_dims(image_resized.astype(np.float32) / 255.0, axis=0)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Retrieve the bounding box and confidence score outputs
boxes = interpreter.get_tensor(output_details[0]['index'])  # Bounding boxes
scores = interpreter.get_tensor(output_details[1]['index'])  # Confidence scores

# Debug: Print shapes and sample contents
print(f"Boxes shape: {boxes.shape}, Output Scores shape: {scores.shape}")
print(f"Sample Box: {boxes[0][0]}")
print(f"Sample Score: {scores[0][0]}")

# Adjust scores to remove the last dimension
scores = scores.squeeze(axis=-1)  # Now scores is of shape (1, 896)

# Display output bounding boxes and scores
for i in range(len(boxes[0])):
    box = boxes[0][i]  # Extract a single box

    # Check the length of the box to see if it has more than 4 values
    if len(box) >= 4:
        score = scores[0][i]
        ymin, xmin, ymax, xmax = box[:4]  # Only take the first four values

        if score > 0:
            # Print the coordinates and the score for each detection
            print(f"Detection {i}: Score = {score}, Box = {box[:4]}")

        # Check if the score is high enough to consider it a valid face detection
        if score > 0.5:
            (left, right, top, bottom) = (xmin * image.shape[1], xmax * image.shape[1],
                                          ymin * image.shape[0], ymax * image.shape[0])

            # Draw bounding box and label only for valid detections
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = f"Face: {score:.2f}"
            cv2.putText(image, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the output image with bounding boxes
cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
