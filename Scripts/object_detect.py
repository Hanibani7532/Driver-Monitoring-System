import cv2
from ultralytics import YOLO

# YOLOv8 Nano model load karein (Ye boht fast hai)
model = YOLO('yolov8n.pt') 

cap = cv2.VideoCapture(0)

print("Mobile Detection Start Horahi Hai...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # YOLO se detection karein
    # classes=[67] ka matlab hai sirf 'cell phone' detect karo
    results = model(frame, conf=0.5, classes=[67], stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Box draw karein
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Label lagayein
            cv2.putText(frame, "WARNING: MOBILE USAGE", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Distraction Detection Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()