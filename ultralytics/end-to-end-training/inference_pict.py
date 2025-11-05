from ultralytics import YOLO  # type: ignore[import-untyped]
import cv2

# Load your model (pretrained or custom)
model = YOLO("yolo11n.pt")  # or "runs/train/exp/weights/best.pt"

# Path to your input image
input_path = "input.jpg"
output_path = "output.jpg"

# Run inference
results = model(input_path)

# Read the original image
frame = cv2.imread(input_path)
if frame is None:
    raise ValueError("Could not load image.jpg")

# Get detections
boxes = results[0].boxes

for box in boxes:
    # Extract coordinates, confidence, and class
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    cls = int(box.cls[0])
    label = model.names[cls]

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label with confidence
    text = f"{label} {conf:.2f}"
    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Write the output image
cv2.imwrite(output_path, frame)
print(f"✅ Saved annotated result to {output_path}")
