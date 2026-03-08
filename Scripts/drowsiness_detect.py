import cv2
import mediapipe as mp
import time
from scipy.spatial import distance as dist

# Standard EAR function using 6 specific landmarks
def calculate_ear(eye_landmarks, frame_w, frame_h):
    coords = []
    # Landmarks ko pixel coordinates mein convert karein
    for lm in eye_landmarks:
        coords.append((int(lm.x * frame_w), int(lm.y * frame_h)))
    
    # EAR Formula: (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    # Vertical distances
    v1 = dist.euclidean(coords[1], coords[5]) # p2, p6
    v2 = dist.euclidean(coords[2], coords[4]) # p3, p5
    # Horizontal distance
    h1 = dist.euclidean(coords[0], coords[3]) # p1, p4
    
    ear = (v1 + v2) / (2.0 * h1)
    return ear

# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Standard 6-point indices for EAR
L_EYE = [362, 385, 387, 263, 373, 380] 
R_EYE = [33, 160, 158, 133, 153, 144]

# Thresholds (Inhe aap apni value dekh kar mazeed adjust kar sakte hain)
EAR_THRESHOLD = 0.23 
CLOSED_TIME_LIMIT = 2.0 
eye_closed_start_time = 0

cap = cv2.VideoCapture(0)

print("System Start... Testing EAR range.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_coords = results.multi_face_landmarks[0].landmark
        
        # EAR calculation for both eyes
        left_ear = calculate_ear([mesh_coords[i] for i in L_EYE], w, h)
        right_ear = calculate_ear([mesh_coords[i] for i in R_EYE], w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        # Debugging: Draw only the 6 points for verification
        for id in L_EYE + R_EYE:
            lm = mesh_coords[id]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 1, (0, 0, 255), -1)

        # UI Display
        if avg_ear < EAR_THRESHOLD:
            if eye_closed_start_time == 0:
                eye_closed_start_time = time.time()
            
            elapsed_time = time.time() - eye_closed_start_time
            if elapsed_time > CLOSED_TIME_LIMIT:
                cv2.putText(frame, "!!! WAKE UP !!!", (150, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        else:
            eye_closed_start_time = 0
            cv2.putText(frame, "STATUS: Active", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # EAR Value display at bottom
        cv2.putText(frame, f"EAR Value: {avg_ear:.2f}", (30, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow('Drowsiness Detection Prototype', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()