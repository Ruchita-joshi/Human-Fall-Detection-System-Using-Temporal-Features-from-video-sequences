# train_temporal_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. LOAD TEMPORAL FEATURES
# -----------------------------
csv_path = "temporal_features_cleaned.csv"
df = pd.read_csv(csv_path)

print("=" * 70)
print("TEMPORAL FEATURE MODEL TRAINING")
print("=" * 70)
print(f"Dataset shape: {df.shape}")
print(f"Features: {df.shape[1] - 5} temporal features")  # minus subject, action, video, start_frame, label

# Check class distribution
print("\nClass Distribution:")
fall_count = df['label'].sum()
normal_count = len(df) - fall_count
print(f"FALL windows: {fall_count} ({fall_count/len(df)*100:.1f}%)")
print(f"NORMAL windows: {normal_count} ({normal_count/len(df)*100:.1f}%)")

# -----------------------------
# 2. PREPARE FEATURES AND LABELS
# -----------------------------
# Drop metadata columns, keep only features
X = df.drop(['subject', 'action', 'video', 'start_frame', 'label'], axis=1).values
y = df['label'].values

# Create video groups for proper cross-validation
video_ids = df['subject'] + "_" + df['action'] + "_" + df['video']
unique_videos = video_ids.unique()
print(f"\nUnique videos: {len(unique_videos)}")

# -----------------------------
# 3. GROUP-WISE SPLIT (CRITICAL!)
# -----------------------------
# Split by video, not by windows
train_videos, test_videos = train_test_split(
    unique_videos, test_size=0.2, random_state=42,
    stratify=df.groupby(video_ids)['label'].first()  # Use first label per video
)

train_mask = video_ids.isin(train_videos)
test_mask = video_ids.isin(test_videos)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
train_video_ids, test_video_ids = video_ids[train_mask], video_ids[test_mask]

print(f"\nTraining set: {len(X_train)} windows from {len(train_videos)} videos")
print(f"Test set: {len(X_test)} windows from {len(test_videos)} videos")

# -----------------------------
# 4. FEATURE SCALING
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. TRAIN RANDOM FOREST (Better for temporal features!)
# -----------------------------
print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST ON TEMPORAL FEATURES")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=150,           # More trees for complex patterns
    max_depth=None,             # Let trees grow deep
    min_samples_split=5,        # Prevent overfitting
    min_samples_leaf=2,
    class_weight='balanced',    # Handle imbalance
    n_jobs=-1,                  # Use all cores
    random_state=42,
    bootstrap=True
)

# Cross-validation with group-wise splitting
print("\nPerforming 5-fold group-wise cross-validation...")
gkf = GroupKFold(n_splits=5)
cv_scores = cross_val_score(
    model, X_train_scaled, y_train,
    cv=gkf.split(X_train_scaled, y_train, train_video_ids),
    scoring='f1_macro',
    n_jobs=-1
)
print(f"CV F1 Scores: {cv_scores}")
print(f"Mean CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Train final model
print("\nTraining final model on full training set...")
model.fit(X_train_scaled, y_train)

# -----------------------------
# 6. EVALUATE ON TEST SET
# -----------------------------
print("\n" + "=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
print(f"Accuracy: {accuracy:.3f}")
print(f"ROC AUC: {roc_auc:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fall']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print(f"True Negatives (Normal correctly identified): {cm[0,0]}")
print(f"False Positives (Normal misclassified as Fall): {cm[0,1]}")
print(f"False Negatives (Fall misclassified as Normal): {cm[1,0]}")
print(f"True Positives (Fall correctly identified): {cm[1,1]}")

# -----------------------------
# 7. ANALYZE FALSE POSITIVES
# -----------------------------
print("\n" + "=" * 70)
print("FALSE POSITIVE ANALYSIS")
print("=" * 70)

fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
if len(fp_indices) > 0:
    print(f"Found {len(fp_indices)} false positives")
    
    # Check which videos have false positives
    fp_videos = test_video_ids.iloc[fp_indices].unique()
    print(f"False positives in {len(fp_videos)}/{len(test_videos)} test videos")
    
    # Look at confidence scores for false positives
    fp_confidences = y_prob[fp_indices]
    print(f"Average confidence for false positives: {np.mean(fp_confidences):.3f}")
    print(f"Max confidence for false positives: {np.max(fp_confidences):.3f}")
    
    # Check if false positives are borderline cases
    borderline_threshold = 0.6
    borderline_fp = np.sum(fp_confidences < borderline_threshold)
    print(f"Borderline false positives (<{borderline_threshold}): {borderline_fp}/{len(fp_indices)}")
else:
    print("No false positives found!")

# -----------------------------
# 8. FEATURE IMPORTANCE
# -----------------------------
print("\n" + "=" * 70)
print("TOP 15 MOST IMPORTANT TEMPORAL FEATURES")
print("=" * 70)

feature_names = df.drop(['subject', 'action', 'video', 'start_frame', 'label'], axis=1).columns
importances = model.feature_importances_
indices = np.argsort(importances)[-15:][::-1]

for i, idx in enumerate(indices[:15]):
    print(f"{i+1:2d}. {feature_names[idx]:30s} : {importances[idx]:.4f}")

# Plot feature importance
plt.figure(figsize=(12, 8))
bars = plt.barh(range(15), importances[indices[:15]])
plt.yticks(range(15), [feature_names[i] for i in indices[:15]])
plt.xlabel('Feature Importance')
plt.title('Top 15 Temporal Features for Fall Detection')
plt.gca().invert_yaxis()

# Color code by feature type
for i, bar in enumerate(bars):
    feature_name = feature_names[indices[i]]
    if 'velocity' in feature_name.lower():
        bar.set_color('red')
    elif 'angle' in feature_name.lower():
        bar.set_color('blue')
    elif 'aspect' in feature_name.lower():
        bar.set_color('green')
    elif 'ground' in feature_name.lower():
        bar.set_color('orange')
    elif 'jerk' in feature_name.lower():
        bar.set_color('purple')

plt.tight_layout()
plt.savefig('temporal_feature_importance.png', dpi=100, bbox_inches='tight')
print("\nFeature importance plot saved to 'temporal_feature_importance.png'")

# -----------------------------
# 9. SAVE MODEL
# -----------------------------
joblib.dump(model, "fall_detection_temporal_rf.pkl")
joblib.dump(scaler, "scaler_temporal.pkl")

print("\n" + "=" * 70)
print("MODEL SAVED SUCCESSFULLY!")
print("=" * 70)
print("Model: fall_detection_temporal_rf.pkl")
print("Scaler: scaler_temporal.pkl")
print("\nKey improvements over single-frame model:")
print("1. Temporal features (velocity, acceleration, trends)")
print("2. Window-based analysis (1-second windows)")
print("3. Group-wise splitting (prevents data leakage)")
print("4. Random Forest (better for temporal patterns)")
print("5. Proper false positive analysis")