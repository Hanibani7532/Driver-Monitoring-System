import cv2
from ultralytics import YOLO

# Load the YOLOv8 Nano model (Optimized for real-time performance)
# 'yolov8n.pt' is used for high-speed inference on standard hardware
model = YOLO('yolov8n.pt') 

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

print("Mobile Distraction Detection: Initialized...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        break

    # Perform inference using YOLOv8
    # 'conf=0.5' filters out weak detections
    # 'classes=[67]' specifically targets 'cell phone' from the COCO dataset
    # 'stream=True' is used for efficient memory management during live video
    results = model(frame, conf=0.5, classes=[67], stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw a red rectangle around the detected mobile device
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Display a warning label above the bounding box
            cv2.putText(frame, "WARNING: MOBILE USAGE", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show the real-time detection output
    cv2.imshow('Distraction Detection Test', frame)
    
    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
