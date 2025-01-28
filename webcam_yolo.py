import cv2
from ultralytics import YOLO
import torch

# Load the YOLOv5 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov5n.pt')

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model(frame)

    # Annotate the frame with the results
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLO Webcam', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
