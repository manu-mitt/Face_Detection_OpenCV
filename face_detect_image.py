import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Load the pre-trained deep learning model
base_dir = os.path.dirname(os.path.abspath(__file__))
modelFile = os.path.join(base_dir, "res10_300x300_ssd_iter_140000.caffemodel")
configFile = os.path.join(base_dir, "deploy.prototxt.txt")

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Load your input image
image_path = os.path.join(base_dir, "images", "test_image.jpg")
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not read the image at {image_path}. Check the file path or extension.")
    exit()
(h, w) = image.shape[:2]

# Prepare the image for deep learning face detector
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 117.0, 123.0))

# Pass the blob through the network and obtain the detections
net.setInput(blob)
detections = net.forward()

# Create directory to store detected faces
os.makedirs(os.path.join(base_dir, "detected_faces"), exist_ok=True)

# Loop over the detections and draw bounding boxes
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections by ensuring confidence > 0.5
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Ensure coordinates are within image boundaries
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(w, endX), min(h, endY)

        # Crop and save the detected face
        face_roi = image[startY:endY, startX:endX]
        cv2.imwrite(os.path.join(base_dir, "detected_faces", f"face_{i}.jpg"), face_roi)

        # Draw bounding box and confidence score
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10

        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# Show the output image
# Convert BGR (OpenCV) to RGB (Matplotlib)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis("off")
plt.show()
