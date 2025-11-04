# infer_yolo11n.py
from ultralytics import YOLO
import cv2
import torch

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Use webcam (0) or video path
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Get annotated frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLO Inference", annotated_frame)

    # Monitor GPU (PyTorch)
    print(f"GPU Used: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
