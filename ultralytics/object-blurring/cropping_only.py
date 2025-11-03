import cv2

from ultralytics import solutions

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera. Check camera index or permissions.")
    exit()

# Initialize object cropper object
cropper = solutions.ObjectCropper(
    show=False,  # display the output
    model="../yolo11n.pt",  # model for object cropping i.e yolo11x.pt.
    classes=[0],  # class
    crop_dir="cropped-detections",  # set the directory name for cropped detections
    # conf=0.5,  # adjust confidence threshold for the objects.
    # crop_dir="cropped-detections",  # set the directory name for cropped detections
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = cropper(im0)
    print(results)
    cv2.imshow("Custom YOLO Stream", im0)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  # destroy all opened windows