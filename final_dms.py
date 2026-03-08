import cv2
import winsound
from ultralytics import YOLO

# Model path check karlein
distraction_model = YOLO('models/trained/distraction_model.pt') 

class_names = ['Safe Driving', 'Texting (R)', 'Talking Phone (R)', 'Texting (L)', 
               'Talking Phone (L)', 'Operating Radio', 'Drinking', 'Reaching Behind', 
               'Hair/Makeup', 'Talking to Passenger']

cap = cv2.VideoCapture(0)

# 🔧 SUPER SENSITIVE TUNING
CONF_LIMIT = 0.60    # Isay 0.40 kar diya taake 40% confidence par bhi alert de
      
print("System Testing... Ab mobile check karein.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = distraction_model(frame, verbose=False)
    
    for r in results:
        class_id = r.probs.top1
        conf = r.probs.top1conf.item()
        label = class_names[class_id]

        # Dangerous classes (Mobile, Drinking, etc.)
        # Hum c0 (Safe) ke ilawa sab par nazar rakhenge
        if class_id != 0 and conf > CONF_LIMIT:
            status_text = f"ALERT: {label}!"
            color = (0, 0, 255) # Red
            winsound.Beep(1000, 100) # Quick Beep
        else:
            status_text = "Safe"
            color = (0, 255, 0) # Green

        # Display results
        cv2.putText(frame, f"{status_text} ({conf:.2f})", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Super Sensitive DMS', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()