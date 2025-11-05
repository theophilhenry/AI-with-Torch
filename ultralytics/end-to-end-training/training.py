# train_yolo11n.py
from ultralytics import YOLO  # type: ignore[import-untyped]

# Load a pretrained YOLO11n model (smallest, fastest)
model = YOLO("yolo11n.pt")

if __name__ == "__main__":
  # Train the model
  model.train(
      data="./data/data.yaml",     # Path to your dataset YAML file
      epochs=50,            # Number of training epochs
      imgsz=640,            # Image size (YOLO will resize automatically)
      batch=-1,             # Auto batch size (YOLO finds best fit for GPU VRAM)
      pretrained=True,      # Start from pretrained COCO weights
      device=0              # Use GPU (0), or 'cpu' if no CUDA
  )
