import cv2
from ultralytics import solutions

# --- 1. Change source to camera (0 is typically the default webcam) ---
# Use 0 for the default camera, or 1, 2, etc., for other cameras.
cap = cv2.VideoCapture(0)

# Check if the camera was opened successfully
if not cap.isOpened():
    print("Error: Could not open camera. Check camera index or permissions.")
    exit()

# It's good practice to set a resolution if your camera supports it, 
# but we'll try to read the properties first.
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Camera opened with resolution: {w}x{h} at {fps:.2f} FPS")

# Define region points based on the camera resolution (adjust as needed for your view)
# Using a central region for demonstration (e.g., a central square)
region_points = [
    (int(w * 0.2), int(h * 0.2)),
    (int(w * 0.8), int(h * 0.2)),
    (int(w * 0.8), int(h * 0.8)),
    (int(w * 0.2), int(h * 0.8))
]

# Init trackzone 
# Setting 'show=True' is crucial for displaying the stream using internal Ultralytics/OpenCV logic
trackzone = solutions.TrackZone(
    show=False,              # Display the output in a window
    region=region_points,   # Pass region points
    model="yolov8n.pt",     # Using a more standard YOLOv8 nano model
    line_width=2            # Optional: adjust line thickness
)

# Process live stream
while cap.isOpened():
    success, im0 = cap.read()
    
    if not success:
        print("Camera stream ended or failed to read frame.")
        # Attempt to reconnect or break the loop
        break
    
    # Process the frame with the TrackZone solution
    # The 'trackzone' object internally calls the model and plots the results onto 'im0'.
    # Because 'show=True' was set during initialization, Ultralytics usually handles 
    # the display internally via cv2.imshow().
    results = trackzone(im0)
    
    # If the internal 'show=True' feature is not working or you want more control,
    # you can uncomment the following line and handle the display manually:
    cv2.putText(results.plot_im, "Hello World", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Live YOLO TrackZone Stream", results.plot_im)

    # --- 2. Live Stream Output: Use cv2.waitKey(1) to keep the window updated ---
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
