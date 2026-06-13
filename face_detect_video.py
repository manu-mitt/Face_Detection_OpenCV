import cv2
import numpy as np
import os

# Load model files
base_dir = os.path.dirname(os.path.abspath(__file__))
modelFile = os.path.join(base_dir, "res10_300x300_ssd_iter_140000.caffemodel")
configFile = os.path.join(base_dir, "deploy.prototxt.txt")

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Performance: Use OpenCL for hardware acceleration if available
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Open your video file here (change the path if needed)
cap = cv2.VideoCapture(0)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
skip_frames = 2  # Run detection every 2nd frame
detections = None

# Create directory to store detected faces from video
output_dir = os.path.join(base_dir, "detected_faces_video")
os.makedirs(output_dir, exist_ok=True)

while True:
    ret, frame = cap.read()

    if not ret:
        # No more frames left (video ended)
        print("Reached end of video, exiting...")
        break

    (h, w) = frame.shape[:2]

    # Speed up: Only run detection every 'skip_frames'
    if frame_count % skip_frames == 0:
        # Pre-process frame (removed redundant manual resize)
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 117.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # Save detected faces as individual files
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # Coordinate boundary checks
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                
                face_roi = frame[startY:endY, startX:endX]
                if face_roi.size > 0:
                    cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count}_face_{i}.jpg"), face_roi)
    
    frame_count += 1

    # Draw boxes using the most recent detections
    if detections is not None:
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Coordinate boundary checks for drawing
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)

                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10

                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # UI Optimization: Resize frame for faster display if resolution is high
    display_frame = frame
    if w > 1280:
        scale = 1280.0 / w
        display_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    # Show the video frame with detections
    try:
        cv2.imshow("Face Detection", display_frame)
        
        # Wait 1 millisecond for key press
        # If 'q' is pressed, exit the loop early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("You pressed 'q', exiting...")
            break
    except cv2.error:
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames... (GUI unavailable, but faces are being saved to {output_dir})")
        # In headless mode, we can't use waitKey, so we rely on Ctrl+C to stop
        pass

# Release video and close windows
cap.release()
cv2.destroyAllWindows()
