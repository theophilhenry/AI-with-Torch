# train_yolo11n.py
from ultralytics import YOLO
import os
import threading
import time
import subprocess

# === GPU MONITORING THREAD ===
def monitor_gpu():
    """Show GPU usage live using nvidia-smi every 2 seconds."""
    while True:
        try:
            os.system('clear')
            print("=== Live GPU Monitor (press Ctrl+C to stop training) ===\n")
            subprocess.run(["nvidia-smi"])
            time.sleep(2)
        except KeyboardInterrupt:
            break

# === TRAINING ===
def train_model():
    # Load pretrained YOLO11n model (from COCO)
    model = YOLO("yolo11n.pt")

    # Train
    model.train(
        data="data.yaml",          # your dataset YAML
        epochs=50,                 # number of epochs
        batch=-1,                  # auto batch size
        imgsz=640,                 # image size
        pretrained=True,           # start from pretrained weights
        device=0                   # GPU 0
    )

if __name__ == "__main__":
    # Run monitoring in background thread
    monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
    monitor_thread.start()

    # Start training
    train_model()
