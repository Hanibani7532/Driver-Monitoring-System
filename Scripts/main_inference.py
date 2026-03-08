import cv2
import mediapipe as mp
import time
from scipy.spatial import distance as dist
from ultralytics import YOLO
import winsound  # Library for system alarm (Windows default)

# --- Functions ---

def calculate_ear(eye_landmarks, frame_w, frame_h):
    """Calculate the Eye Aspect Ratio (EAR) using specific facial landmarks."""
    coords = []
    # Convert normalized landmarks to pixel coordinates
    for lm in eye_landmarks:
        coords.append((int(lm.x * frame_w), int(lm.y * frame_h)))
    
    # EAR Formula: vertical distances / (2 * horizontal distance)
    v1 = dist.euclidean(coords[1], coords[5])
    v2 = dist.euclidean(coords[2], coords[4])
    h1 = dist.euclidean(coords[0], coords[3])
    
    return (v1 + v2) / (2.0 * h1)

def play_alarm():
    """Trigger a high-frequency beep sound as a warning alert."""
    # Frequency: 2500Hz, Duration: 500ms
    winsound.Beep(2500, 500)

# --- System Setup ---

# Load pre-trained YOLOv8 nano model for object detection
model = YOLO('yolov8n.pt') 

# Initialize MediaPipe Face Mesh for eye tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Standard 6-point landmark indices for EAR calculation
L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [33, 160, 158, 133, 153, 144]

# Threshold Settings
EAR_THRESHOLD = 0.23      # EAR value below this is considered 'eyes closed'
CLOSED_TIME_LIMIT = 2.0   # Maximum time in seconds before drowsiness alert
eye_closed_start_time = 0

# Start Video Capture
cap = cv2.VideoCapture(0)

print("Driver Monitoring System: Initialized...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        break
        
    h, w, _ = frame.shape
    
    # 1. AI Drowsiness Monitoring (Face Mesh Processing)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)
    
    is_drowsy = False
    if face_results.multi_face_landmarks:
        mesh_coords = face_results.multi_face_landmarks[0].landmark
        
        # Calculate EAR for both eyes
        left_ear = calculate_ear([mesh_coords[i] for i in L_EYE], w, h)
        right_ear = calculate_ear([mesh_coords[i] for i in R_EYE], w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        # Drowsiness Logic
        if avg_ear < EAR_THRESHOLD:
            if eye_closed_start_time == 0: 
                eye_closed_start_time = time.time()
            # Trigger alert if eyes stay closed longer than the time limit
            if time.time() - eye_closed_start_time > CLOSED_TIME_LIMIT:
                is_drowsy = True
        else:
            eye_closed_start_time = 0

    # 2. Distraction Detection (YOLOv8 Phone Detection)
    # class 67 is for 'cell phone' in the COCO dataset
    yolo_results = model(frame, conf=0.5, classes=[67], verbose=False) 
    
    phone_detected = False
    for r in yolo_results:
        if len(r.boxes) > 0:
            phone_detected = True
            for box in r.boxes:
                # Draw bounding box for visual alert
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # 3. Decision Making & UI Feedback
    if is_drowsy or phone_detected:
        status_msg = "!!! ALERT: DRIVER DISTRACTED !!!"
        color = (0, 0, 255) # Red for alert
        play_alarm()       # Trigger system alert
    else:
        status_msg = "STATUS: Safe Driving"
        color = (0, 255, 0) # Green for safe status

    # Render on-screen status message
    cv2.putText(frame, status_msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    # Display the final inference result
    cv2.imshow('Driver AI Monitoring System', frame)
    
    # Press 'q' to terminate the program
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Release camera and clear windows
cap.release()
cv2.destroyAllWindows()
