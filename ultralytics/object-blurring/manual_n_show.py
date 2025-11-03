import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# --- Camera Setup (Remains the same) ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera. Check camera index or permissions.")
    exit()

model = YOLO("../yolov8n.pt")
class_names = model.names

# Process live stream
while cap.isOpened():
    success, im0 = cap.read()
    
    if not success:
        print("Camera stream ended or failed to read frame.")
        break

    results = model(im0)
    boxes = results[0].boxes
    annotator = Annotator(im0, line_width=2, example=class_names)
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        conf = float(box.conf)
        label = class_names[int(box.cls)]

        annotator.box_label(box.xyxy[0], color=(0, 0, 255), label=label)
        cropped_image = im0[y1:y2, x1:x2]
        # blurred_image = cv2.GaussianBlur(cropped_image, (15, 15), 0)
        blurred_image = cv2.blur(cropped_image, (50, 50))
        im0[y1:y2, x1:x2] = blurred_image

    cv2.imshow("Custom YOLO Stream", im0)

    # --- Live Stream Output: Use cv2.waitKey(1) to keep the window updated ---
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()