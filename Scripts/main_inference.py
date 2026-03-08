import cv2
import mediapipe as mp
import time
from scipy.spatial import distance as dist
from ultralytics import YOLO
import winsound  # Alarm bajane ke liye (Windows default)

# --- Functions ---
def calculate_ear(eye_landmarks, frame_w, frame_h):
    coords = []
    for lm in eye_landmarks:
        coords.append((int(lm.x * frame_w), int(lm.y * frame_h)))
    v1 = dist.euclidean(coords[1], coords[5])
    v2 = dist.euclidean(coords[2], coords[4])
    h1 = dist.euclidean(coords[0], coords[3])
    return (v1 + v2) / (2.0 * h1)

def play_alarm():
    # Frequency 2500Hz, Duration 500ms
    winsound.Beep(2500, 500)

# --- Setup ---
model = YOLO('yolov8n.pt') 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.23
CLOSED_TIME_LIMIT = 2.0
eye_closed_start_time = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    h, w, _ = frame.shape
    
    # 1. AI Monitoring (Face Mesh)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)
    
    is_drowsy = False
    if face_results.multi_face_landmarks:
        mesh_coords = face_results.multi_face_landmarks[0].landmark
        left_ear = calculate_ear([mesh_coords[i] for i in L_EYE], w, h)
        right_ear = calculate_ear([mesh_coords[i] for i in R_EYE], w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            if eye_closed_start_time == 0: eye_closed_start_time = time.time()
            if time.time() - eye_closed_start_time > CLOSED_TIME_LIMIT:
                is_drowsy = True
        else:
            eye_closed_start_time = 0

    # 2. Object Detection (Phone)
    yolo_results = model(frame, conf=0.5, classes=[67], verbose=False) # 67 = Cell phone
    phone_detected = False
    for r in yolo_results:
        if len(r.boxes) > 0:
            phone_detected = True
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # 3. Decision & Alarm
    if is_drowsy or phone_detected:
        status_msg = "!!! ALERT: DRIVER DISTRACTED !!!"
        color = (0, 0, 255) # Red
        play_alarm() # Beep sound
    else:
        status_msg = "STATUS: Safe Driving"
        color = (0, 255, 0) # Green

    cv2.putText(frame, status_msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    cv2.imshow('Driver AI Monitoring System', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()