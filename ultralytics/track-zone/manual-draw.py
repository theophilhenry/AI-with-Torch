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
    (int(w * 0.2), int(h * 0.2)),
    (int(w * 0.8), int(h * 0.2)),
    (int(w * 0.8), int(h * 0.8)),
    (int(w * 0.2), int(h * 0.8))
]

# --- TrackZone Initialization (Crucial Change) ---
# Set 'show=False' to stop the solution from plotting automatically.
# We will manually plot the results on the frame.
trackzone = solutions.TrackZone(
    show=False,              
    region=region_points,   
    model="yolov8n.pt",     
    line_width=2            
)

# Process live stream
while cap.isOpened():
    success, im0 = cap.read()
    
    if not success:
        print("Camera stream ended or failed to read frame.")
        break
    
    # Process the frame with the TrackZone solution
    # The 'results' object contains all the detection and tracking data.
    # Note: 'im0' is modified internally by TrackZone to draw the zone itself, 
    # but the detections/tracks are returned in 'results'.
    results = trackzone(im0)
    # im0 will automatically be modified by TrackZone to draw the zone itself, 
    # but the detections/tracks are returned in 'results'.
    print(results.total_tracks)


    # # --- Manual Drawing Logic (The core customization) ---
    
    # # Check if there are any results to process
    # if results.boxes is not None and len(results.boxes) > 0:
    #     # 'results.boxes' contains the bounding box data.
    #     # It's an ultralytics.engine.results.Boxes object.
        
    #     # 'results.names' is the dictionary mapping class IDs to names (e.g., {0: 'person'}).
    #     class_names = results.names
        
    #     # Loop through each detected bounding box
    #     for box in results.boxes:
    #         # Get the coordinates (normalized to the original image size)
    #         # xyxy is the format [x_min, y_min, x_max, y_max]
    #         x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
            
    #         # Get the confidence score (float)
    #         conf = box.conf[0].item() if box.conf is not None else 0.0
            
    #         # Get the class ID (int) and corresponding label (string)
    #         cls = int(box.cls[0].item())
    #         label = class_names[cls]
            
    #         # --- 1. Draw the Bounding Box (Rectangle) ---
    #         color = (0, 255, 0) # Green for the box
    #         cv2.rectangle(im0, (x_min, y_min), (x_max, y_max), color, 2)
            
    #         # --- 2. Create the label text ---
    #         # Format: 'label conf%' e.g., 'person 0.95'
    #         text = f'{label} {conf:.2f}'
            
    #         # --- 3. Draw the Background for the Label (Optional but good practice) ---
    #         # This makes the text readable against a busy background.
    #         (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    #         cv2.rectangle(im0, (x_min, y_min - text_h - baseline), (x_min + text_w, y_min), color, -1) # -1 fills the rectangle
            
    #         # --- 4. Draw the Label Text (Confidence and Class) ---
    #         # Text position: top-left corner of the bounding box
    #         text_color = (0, 0, 0) # Black text on the green background
    #         cv2.putText(im0, text, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)


    # The 'im0' frame now contains the original image PLUS the region line (from TrackZone) 
    # PLUS the manually drawn bounding boxes and labels.
    cv2.imshow("Custom YOLO Stream", im0)

    # --- Live Stream Output: Use cv2.waitKey(1) to keep the window updated ---
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()