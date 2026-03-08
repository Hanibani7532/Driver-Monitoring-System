import cv2
import winsound
from ultralytics import YOLO

# Load the custom trained distraction classification model
# Path: models/trained/distraction_model.pt
distraction_model = YOLO('models/trained/distraction_model.pt') 

# Define class labels for the 10 distraction categories (c0 to c9)
class_names = ['Safe Driving', 'Texting (R)', 'Talking Phone (R)', 'Texting (L)', 
               'Talking Phone (L)', 'Operating Radio', 'Drinking', 'Reaching Behind', 
               'Hair/Makeup', 'Talking to Passenger']

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# --- SENSITIVITY CONFIGURATION ---
# Set the confidence threshold (Lowering this value makes the system more sensitive)
CONF_LIMIT = 0.60 

print("System Testing: Monitoring for distractions and mobile usage...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        break

    # Perform inference on the current frame
    results = distraction_model(frame, verbose=False)
    
    for r in results:
        # Get the top predicted class and its confidence score
        class_id = r.probs.top1
        conf = r.probs.top1conf.item()
        label = class_names[class_id]

        # Alert Logic: Monitor all classes except 'c0' (Safe Driving)
        # Trigger alarm if confidence exceeds the set threshold
        if class_id != 0 and conf > CONF_LIMIT:
            status_text = f"ALERT: {label}!"
            color = (0, 0, 255) # Red for high-risk detection
            winsound.Beep(1000, 100) # Quick auditory warning
        else:
            status_text = "Safe"
            color = (0, 255, 0) # Green for normal state

        # Render the status message and confidence score on screen
        cv2.putText(frame, f"{status_text} ({conf:.2f})", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the live detection window
    cv2.imshow('Super Sensitive DMS', frame)
    
    # Press 'q' to stop the monitoring system
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Release hardware resources
cap.release()
cv2.destroyAllWindows()
