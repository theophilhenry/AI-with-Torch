import cv2
import numpy as np
from ultralytics import solutions

# --- Camera Setup (Remains the same) ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera. Check camera index or permissions.")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS) # Not strictly needed for the loop

# Define region points based on the camera resolution
region_points = [
    (int(w * 0.45), 0),
    (int(w * 0.45), int(h)),
    (int(w * 0.55), int(h)),
    (int(w * 0.55), int(0)),
]
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]
counter = solutions.ObjectCounter(
  show=True, 
  region=region_points, 
  model="../yolov8n.pt",
  # conf=0.8,
  classes=[0],
  show_labels=False,
)

# Process live stream
while cap.isOpened():
    success, im0 = cap.read()
    
    if not success:
        print("Camera stream ended or failed to read frame.")
        break

    results = counter(im0)
    print(results.total_tracks)

    cv2.imshow("Custom YOLO Stream", im0)

    # --- Live Stream Output: Use cv2.waitKey(1) to keep the window updated ---
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()