# extract_temporal_features.py
import cv2
import mediapipe as mp
import os
import csv
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Root dataset path
dataset_path = r"C:\Users\ruchi\Downloads\GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1\ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"

output_csv = "temporal_features.csv"

subjects = ["Subject 1", "Subject 2", "Subject 3", "Subject 4"]
actions = ["ADL", "Fall"]

# Helper functions for temporal feature extraction
def extract_window_features(window_data):
    """Extract statistical features from a window of frames"""
    features = []
    
    # Statistical moments for each coordinate type
    for coord_idx in range(4):  # x, y, z, visibility
        coord_data = window_data[:, coord_idx::4]  # Get all x's, then all y's, etc.
        
        # Mean and std for each landmark
        means = np.mean(coord_data, axis=0)
        stds = np.std(coord_data, axis=0)
        
        # Global statistics
        features.append(np.mean(means))  # Global mean
        features.append(np.std(means))   # Variation across landmarks
        features.append(np.mean(stds))   # Average movement
        features.append(np.max(stds))    # Maximum movement
    
    # Velocity features (change between consecutive frames)
    velocities = []
    for i in range(1, len(window_data)):
        # Calculate Euclidean distance between frames (using x,y only)
        prev_frame = window_data[i-1].reshape(-1, 4)[:, :2]  # x,y only
        curr_frame = window_data[i].reshape(-1, 4)[:, :2]
        vel = np.mean(np.sqrt(np.sum((curr_frame - prev_frame)**2, axis=1)))
        velocities.append(vel)
    
    if velocities:
        features.append(np.mean(velocities))  # Average velocity
        features.append(np.max(velocities))   # Peak velocity
        features.append(np.std(velocities))   # Velocity variation
    
    # Key body part features
    window_reshaped = window_data.reshape(-1, 33, 4)
    
    # Torso angle variation
    angles = []
    for frame in window_reshaped:
        # Shoulder center
        left_shoulder = frame[11][:2]
        right_shoulder = frame[12][:2]
        shoulder_center = (left_shoulder + right_shoulder) / 2
        
        # Hip center
        left_hip = frame[23][:2]
        right_hip = frame[24][:2]
        hip_center = (left_hip + right_hip) / 2
        
        # Torso vector and angle
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
    
    # Aspect ratio (height/width)
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
    
    # Ground contact (ankles near bottom)
    ground_contacts = []
    for frame in window_reshaped:
        left_ankle_y = frame[27][1]
        right_ankle_y = frame[28][1]
        ground_contacts.append(max(left_ankle_y, right_ankle_y))
    
    if ground_contacts:
        features.append(np.mean(ground_contacts))
        features.append(np.min(ground_contacts))
    
    # Movement smoothness (jerk)
    if len(window_data) >= 3:
        # Calculate acceleration changes
        centroids = []
        for frame in window_reshaped:
            centroids.append(np.mean(frame[:, :2], axis=0))
        
        centroids = np.array(centroids)
        velocities = np.sqrt(np.sum(np.diff(centroids, axis=0)**2, axis=1))
        if len(velocities) >= 2:
            accelerations = np.diff(velocities)
            if len(accelerations) >= 1:
                jerk = np.std(accelerations) if len(accelerations) > 0 else 0
                features.append(jerk)
    
    return features

# Create CSV
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Create header (we'll add it after knowing feature count)
    header_written = False
    
    # Loop through subject and action folders
    for sub in subjects:
        for act in actions:
            folder_path = os.path.join(dataset_path, sub, act)
            label = 1 if act == "Fall" else 0
            
            if not os.path.exists(folder_path):
                print("❌ Folder NOT found:", folder_path)
                continue

            for video_file in os.listdir(folder_path):
                video_path = os.path.join(folder_path, video_file)
                
                if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
                    continue
                
                print(f"Processing: {sub}/{act}/{video_file}")
                cap = cv2.VideoCapture(video_path)
                
                # Collect all frames first
                all_frames_data = []
                frame_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every frame (no skipping)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)
                    
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        row = []
                        for lm in landmarks:
                            row += [lm.x, lm.y, lm.z, lm.visibility]
                        all_frames_data.append(row)
                    else:
                        # If no detection, use zeros
                        all_frames_data.append([0]*132)  # 33 landmarks * 4
                    
                    frame_count += 1
                
                cap.release()
                
                if len(all_frames_data) < 30:  # Skip if video too short
                    print(f"  Skipped: Only {len(all_frames_data)} frames")
                    continue
                
                # Convert to numpy
                all_frames_data = np.array(all_frames_data)
                
                # Create sliding windows (30 frames = 1 second at 30fps)
                window_size = 30
                step_size = 15  # 50% overlap
                
                for start_idx in range(0, len(all_frames_data) - window_size + 1, step_size):
                    window = all_frames_data[start_idx:start_idx + window_size]
                    
                    # Extract temporal features from window
                    window_features = extract_window_features(window)
                    
                    # Add video info and label
                    video_info = [sub, act, video_file, start_idx]
                    final_features = video_info + window_features + [label]
                    
                    # Write header if not written
                    if not header_written:
                        # Create feature names
                        header = ['subject', 'action', 'video', 'start_frame']
                        # Add feature names
                        feature_names = []
                        # Mean features
                        for coord in ['x', 'y', 'z', 'v']:
                            feature_names.extend([
                                f'{coord}_global_mean',
                                f'{coord}_mean_variation',
                                f'{coord}_avg_movement',
                                f'{coord}_max_movement'
                            ])
                        # Velocity features
                        feature_names.extend(['avg_velocity', 'max_velocity', 'velocity_std'])
                        # Angle features
                        feature_names.extend(['mean_angle', 'angle_std', 'max_angle'])
                        # Aspect ratio
                        feature_names.extend(['mean_aspect', 'aspect_std'])
                        # Ground contact
                        feature_names.extend(['mean_ground_contact', 'min_ground_contact'])
                        # Jerk
                        feature_names.append('movement_jerk')
                        
                        header = header + feature_names + ['label']
                        writer.writerow(header)
                        header_written = True
                    
                    writer.writerow(final_features)
                
                print(f"  Extracted {((len(all_frames_data) - window_size) // step_size) + 1} windows")

print(f"\n✅ Temporal feature extraction completed! Saved to {output_csv}")
print(f"   Each window = 30 frames (~1 second)")
print(f"   Features represent movement patterns over time, not just static poses")