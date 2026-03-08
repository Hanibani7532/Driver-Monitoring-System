import cv2
import mediapipe as mp
import time
from scipy.spatial import distance as dist

# Standard EAR (Eye Aspect Ratio) function using 6 specific landmarks
def calculate_ear(eye_landmarks, frame_w, frame_h):
    coords = []
    # Convert landmarks to pixel coordinates
    for lm in eye_landmarks:
        coords.append((int(lm.x * frame_w), int(lm.y * frame_h)))
    
    # EAR Formula: (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    # Vertical distances
    v1 = dist.euclidean(coords[1], coords[5]) # distance between p2 and p6
    v2 = dist.euclidean(coords[2], coords[4]) # distance between p3 and p5
    # Horizontal distance
    h1 = dist.euclidean(coords[0], coords[3]) # distance between p1 and p4
    
    # Calculate final EAR value
    ear = (v1 + v2) / (2.0 * h1)
    return ear

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Standard 6-point landmark indices for EAR calculation
# Left Eye and Right Eye landmark points
L_EYE = [362, 385, 387, 263, 373, 380] 
R_EYE = [33, 160, 158, 133, 153, 144]

# Detection Thresholds (Adjust these values based on your requirements)
EAR_THRESHOLD = 0.23      # Minimum EAR value to consider eyes as open
CLOSED_TIME_LIMIT = 2.0   # Maximum time (seconds) eyes can remain closed before alarm
eye_closed_start_time = 0

# Initialize Webcam
cap = cv2.VideoCapture(0)

print("System Starting... Analyzing EAR range.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        break
    
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_coords = results.multi_face_landmarks[0].landmark
        
        # Calculate EAR for both eyes and get the average
        left_ear = calculate_ear([mesh_coords[i] for i in L_EYE], w, h)
        right_ear = calculate_ear([mesh_coords[i] for i in R_EYE], w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        # Debugging: Draw the 6 EAR points on the frame for visual verification
        for id in L_EYE + R_EYE:
            lm = mesh_coords[id]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 1, (0, 0, 255), -1)

        # UI Logic for Drowsiness Detection
        if avg_ear < EAR_THRESHOLD:
            # Start timer if eyes are detected as closed
            if eye_closed_start_time == 0:
                eye_closed_start_time = time.time()
            
            # Check if time limit has been exceeded
            elapsed_time = time.time() - eye_closed_start_time
            if elapsed_time > CLOSED_TIME_LIMIT:
                cv2.putText(frame, "!!! WAKE UP !!!", (150, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        else:
            # Reset timer when eyes are open
            eye_closed_start_time = 0
            cv2.putText(frame, "STATUS: Active", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display real-time EAR value at the bottom of the screen
        cv2.putText(frame, f"EAR Value: {avg_ear:.2f}", (30, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Show the final output window
    cv2.imshow('Drowsiness Detection Prototype', frame)
    
    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
