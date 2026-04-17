# main_temporal.py
import cv2
import mediapipe as mp
import numpy as np
import argparse
import joblib
from collections import deque
import time
import winsound
from email_alert import send_email_alert

# ---------------- ALERT SOUND ----------------
def play_alert_sound():
    winsound.Beep(1000, 700)

# ---------------- LOAD TEMPORAL MODEL ----------------
try:
    model = joblib.load("fall_detection_temporal_rf.pkl")
    scaler = joblib.load("scaler_temporal.pkl")
    print("Temporal model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading temporal model: {e}")
    print("   Make sure you've run train_temporal_model.py first!")
    exit()

# ---------------- MEDIAPIPE SETUP ----------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------------- TEMPORAL WINDOW CONFIG ----------------
WINDOW_SIZE = 30  # 30 frames = 1 second at 30fps (must match training)
landmark_buffer = deque(maxlen=WINDOW_SIZE)
prediction_buffer = deque(maxlen=5)  # Smooth final predictions

# ---------------- TEMPORAL FEATURE EXTRACTION (Same as training) ----------------
def extract_window_features(window_data):
    """Same function as in extract_temporal_features.py"""
    if len(window_data) < WINDOW_SIZE:
        return None
    
    features = []
    window_data = np.array(window_data)
    
    # Statistical moments for each coordinate
    for coord_idx in range(4):
        coord_data = window_data[:, coord_idx::4]
        means = np.mean(coord_data, axis=0)
        stds = np.std(coord_data, axis=0)
        
        features.append(np.mean(means))
        features.append(np.std(means))
        features.append(np.mean(stds))
        features.append(np.max(stds))
    
    # Velocity features
    velocities = []
    for i in range(1, len(window_data)):
        prev_frame = window_data[i-1].reshape(-1, 4)[:, :2]
        curr_frame = window_data[i].reshape(-1, 4)[:, :2]
        vel = np.mean(np.sqrt(np.sum((curr_frame - prev_frame)**2, axis=1)))
        velocities.append(vel)
    
    if velocities:
        features.append(np.mean(velocities))
        features.append(np.max(velocities))
        features.append(np.std(velocities))
    else:
        features.extend([0, 0, 0])
    
    # Torso angle features
    window_reshaped = window_data.reshape(-1, 33, 4)
    angles = []
    
    for frame in window_reshaped:
        left_shoulder = frame[11][:2]
        right_shoulder = frame[12][:2]
        shoulder_center = (left_shoulder + right_shoulder) / 2
        
        left_hip = frame[23][:2]
        right_hip = frame[24][:2]
        hip_center = (left_hip + right_hip) / 2
        
        torso_vector = shoulder_center - hip_center
        vertical = np.array([0, -1])
        if np.linalg.norm(torso_vector) > 0.01:
            cos_angle = np.dot(torso_vector, vertical) / (np.linalg.norm(torso_vector) * np.linalg.norm(vertical))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            angles.append(angle)
    
    if angles:
        features.append(np.mean(angles))
        features.append(np.std(angles))
        features.append(np.max(angles))
    else:
        features.extend([0, 0, 0])
    
    # Aspect ratio
    aspect_ratios = []
    for frame in window_reshaped:
        x_coords = frame[:, 0]
        y_coords = frame[:, 1]
        if len(x_coords) > 0 and len(y_coords) > 0:
            height = np.max(y_coords) - np.min(y_coords)
            width = np.max(x_coords) - np.min(x_coords)
            if width > 0.01:
                aspect_ratios.append(height / width)
    
    if aspect_ratios:
        features.append(np.mean(aspect_ratios))
        features.append(np.std(aspect_ratios))
    else:
        features.extend([0, 0])
    
    # Ground contact
    ground_contacts = []
    for frame in window_reshaped:
        left_ankle_y = frame[27][1]
        right_ankle_y = frame[28][1]
        ground_contacts.append(max(left_ankle_y, right_ankle_y))
    
    if ground_contacts:
        features.append(np.mean(ground_contacts))
        features.append(np.min(ground_contacts))
    else:
        features.extend([0, 0])
    
    # Movement smoothness (jerk)
    if len(window_data) >= 3:
        centroids = []
        for frame in window_reshaped:
            centroids.append(np.mean(frame[:, :2], axis=0))
        
        centroids = np.array(centroids)
        velocities = np.sqrt(np.sum(np.diff(centroids, axis=0)**2, axis=1))
        if len(velocities) >= 2:
            accelerations = np.diff(velocities)
            if len(accelerations) >= 1:
                jerk = np.std(accelerations)
                features.append(jerk)
            else:
                features.append(0)
        else:
            features.append(0)
    else:
        features.append(0)
    
    return np.array(features)

# ---------------- LANDMARK EXTRACTION ----------------
def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if not results.pose_landmarks:
        return None, results

    row = []
    for lm in results.pose_landmarks.landmark:
        row += [lm.x, lm.y, lm.z, lm.visibility]

    return row, results

# ---------------- MAIN LOOP ----------------
def run(video_path=None):
    cap = cv2.VideoCapture(0) if video_path is None else cv2.VideoCapture(video_path)
    
    frame_count = 0
    WARMUP_FRAMES = WINDOW_SIZE + 10
    alarm_active = False
    alarm_start_time = 0
    total_falls = 0
    confidence = 0.0
    
    print("\n" + "=" * 60)
    print("TEMPORAL FALL DETECTION SYSTEM")
    print("=" * 60)
    print("• Analyzing movement patterns over time")
    print(f"• Window size: {WINDOW_SIZE} frames (~1 second)")
    print("• Features: Velocity, Acceleration, Angles, Trends")
    print("• Model: Random Forest on temporal features")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video finished.")
            break

        frame_count += 1

        # ------------ WARMUP ------------
        if frame_count <= WARMUP_FRAMES:
            buffer_fill = len(landmark_buffer)
            cv2.putText(frame, f"Filling temporal buffer... {buffer_fill}/{WINDOW_SIZE}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
            cv2.imshow("Temporal Fall Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ----------- PROCESS FRAME -----------
        landmarks, results = extract_landmarks(frame)

        # Draw Pose Skeleton
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Default state
        final_status = "NORMAL"
        color = (0, 255, 0)
        status_details = ""
        
        if landmarks is not None:
            # Add to temporal buffer
            landmark_buffer.append(landmarks)
            
            # Check if we have a full window
            if len(landmark_buffer) == WINDOW_SIZE:
                # Extract temporal features
                temporal_features = extract_window_features(landmark_buffer)
                
                if temporal_features is not None:
                    # Scale and predict
                    features_scaled = scaler.transform(temporal_features.reshape(1, -1))
                    prediction = model.predict(features_scaled)[0]
                    prob = model.predict_proba(features_scaled)[0][1]
                    
                    # Smooth predictions
                    prediction_buffer.append(prediction)
                    
                    if len(prediction_buffer) == prediction_buffer.maxlen:
                        fall_votes = sum(prediction_buffer)
                        
                        if fall_votes >= 4:  # 4 out of 5 windows
                            final_status = "FALL"
                            color = (0, 0, 255)
                            confidence = prob
                            status_details = f"Conf: {confidence:.1%}"
                        else:
                            final_status = "NORMAL"
                            color = (0, 255, 0)
                            confidence = 1 - prob
                            status_details = f"Conf: {confidence:.1%}"
                    else:
                        # Buffer not full
                        if prediction == 1:
                            final_status = "POSSIBLE FALL"
                            color = (0, 165, 255)
                            confidence = prob
                        else:
                            final_status = "NORMAL"
                            color = (0, 255, 0)
                            confidence = 1 - prob
                else:
                    final_status = "PROCESSING"
                    color = (255, 255, 0)
            else:
                # Still filling buffer
                buffer_fill = len(landmark_buffer)
                final_status = f"BUFFERING"
                color = (255, 165, 0)
                status_details = f"{buffer_fill}/{WINDOW_SIZE}"
        else:
            final_status = "NO PERSON"
            color = (255, 255, 255)
            landmark_buffer.clear()
            prediction_buffer.clear()

        # ---------- ALARM LOGIC ----------
        if final_status == "FALL":
            if not alarm_active:
                alarm_active = True
                alarm_start_time = time.time()
                total_falls += 1
                print(f"FALL DETECTED! Confidence: {confidence:.1%}")
                play_alert_sound()
                send_email_alert(frame_count)
                

            color = (0, 0, 255)

        # Auto stop alarm after 3 seconds of normal
        if alarm_active and final_status == "NORMAL":
            if time.time() - alarm_start_time > 3:
                alarm_active = False

        # ---------- DISPLAY ----------
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Main status
        status_text = f"Status: {final_status}"
        if status_details:
            status_text += f" ({status_details})"
        
        cv2.putText(frame, status_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Buffer info
        buffer_fill = len(landmark_buffer)
        cv2.putText(frame, f"Temporal Buffer: {buffer_fill}/{WINDOW_SIZE}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        
        # Frame info
        cv2.putText(frame, f"Frame: {frame_count}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Fall count
        cv2.putText(frame, f"Falls: {total_falls}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Visual buffer
        if WINDOW_SIZE > 0:
            bar_width = 200
            bar_height = 8
            fill_width = int((buffer_fill / WINDOW_SIZE) * bar_width)
            
            cv2.rectangle(frame, (20, 160), (20 + bar_width, 160 + bar_height), 
                          (100, 100, 100), -1)
            if buffer_fill > 0:
                bar_color = (0, 200, 200) if buffer_fill < WINDOW_SIZE else (0, 255, 0)
                cv2.rectangle(frame, (20, 160), (20 + fill_width, 160 + bar_height), 
                              bar_color, -1)
        
        # Flashing border for alarm
        if alarm_active and int(time.time() * 2) % 2 == 0:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 10)

        # Instructions
        cv2.putText(frame, "Q: Quit | R: Reset",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Temporal Fall Detection", frame)

        # ---------- KEY HANDLING ----------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            alarm_active = False
            total_falls = 0
            landmark_buffer.clear()
            prediction_buffer.clear()
            print("🔄 Reset Done - Buffers cleared")

    cap.release()
    cv2.destroyAllWindows()

    print("\n Temporal Detection Summary")
    print(f"Frames processed: {frame_count}")
    print(f"Falls detected: {total_falls}")

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file (leave empty for webcam)")
    args = parser.parse_args()
    
    run(args.video)