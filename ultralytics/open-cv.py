from ultralytics import YOLO
import cv2
import time

# Load YOLO model
model = YOLO('yolov8n.pt')

# Open webcam
cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, verbose=False)
    boxes = results[0].boxes  # all detected boxes

    # ---- Manual drawing section ----
    for box in boxes:
        # Extract coordinates
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        conf = float(box.conf)
        cls = int(box.cls)
        label = model.names[cls]

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label text (with confidence)
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, max(y1 - 10, 20)),  # prevent text from going off-screen
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---- FPS counter ----
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-8)  # avoid divide-by-zero
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show camera
    cv2.imshow("YOLOv8 Manual", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
