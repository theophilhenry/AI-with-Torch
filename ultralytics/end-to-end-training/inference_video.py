# infer_video_yolo11n.py
import cv2
import torch
from ultralytics import YOLO
import time

# === CONFIG ===
input_video = "input.mp4"                 # your input video path
output_video = "output_annotated.mp4"     # output video path
model_path = "runs/detect/train/weights/best.pt"  # your trained model

# === LOAD MODEL ===
model = YOLO(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# === READ VIDEO ===
cap = cv2.VideoCapture(input_video)
assert cap.isOpened(), f"Cannot open video: {input_video}"

# Get input video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Create VideoWriter for output
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# === PROCESS FRAMES ===
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Get annotated frame
    annotated_frame = results[0].plot()

    # Write annotated frame to output video
    out.write(annotated_frame)

    # Optional: show progress in console
    frame_count += 1
    if frame_count % 10 == 0:
        elapsed = time.time() - start_time
        print(f"Processed {frame_count} frames ({frame_count/elapsed:.2f} FPS)")
        # Show GPU usage (optional)
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e6
            print(f"GPU memory used: {mem:.1f} MB")

# === CLEANUP ===
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n✅ Done! Saved output video to: {output_video}")
