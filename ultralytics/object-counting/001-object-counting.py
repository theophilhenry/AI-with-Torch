import cv2
from ultralytics import solutions

cap = cv2.VideoCapture(0)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

counter = solutions.RegionCounter(
    show=True,
    region={
      # "region-01": [(0, 0), (0, h/2), (w/2, h/2), (w/2, 0)], 
      "region-01": [(0, 0), (0, h/2), (w/2, h/2), (w/2, 0)], 
    },
    model="yolo11n.pt",
    conf=0.8,
    classes=[0],
    show_labels=False,
    # tracker="botsort.yaml",
)

assert cap.isOpened(), "Error: Could not open video source"

while True:
  # Ret : Boolean, frame : numpy array
  ret, frame = cap.read()
  if not ret:
    break
  results = counter(frame)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

# Cari dataset
# 
# Roboflow
