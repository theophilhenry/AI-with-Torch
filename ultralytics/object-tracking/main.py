import cv2
import numpy as np
from ultralytics import YOLO  # type: ignore[import-untyped]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
  print("Error: Could not open camera. Check camera index or permissions.")
  exit()

yolo = YOLO("../yolo11n.pt")

# Process live stream
while cap.isOpened():
  ret, frame = cap.read()

  if not ret:
    print("Camera stream ended or failed to read frame.")
    break

  results = yolo.track(frame, persist=False)

  for result in results:
    boxes = result.boxes
    for box in boxes:
      x1, y1, x2, y2 = box.xyxy[0].int().tolist()
      conf = float(box.conf)
      cls = int(box.cls)
      label = yolo.names[cls]
      track_id = int(box.id[0]) if box.id is not None else -1

      label = f"ID {track_id} {label} {conf:.2f}"

      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

  cv2.imshow("Custom YOLO Stream", frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
