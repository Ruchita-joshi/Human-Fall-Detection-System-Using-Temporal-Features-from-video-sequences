import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix
)

# -----------------------------
# 1. LOAD CSV FEATURES
# -----------------------------
csv_path = "temporal_features_cleaned.csv"
df = pd.read_csv(csv_path)

print("✅ CSV loaded:", df.shape)

# -----------------------------
# 2. SPLIT FEATURES & LABEL
# -----------------------------
X = df.drop(['subject', 'action', 'video', 'start_frame', 'label'], axis=1)
y = df['label']

# -----------------------------
# 3. LOAD MODEL & SCALER
# -----------------------------
model = joblib.load("fall_detection_temporal_rf.pkl")
scaler = joblib.load("scaler_temporal.pkl")
print("✅ Model loaded (No retraining)")

# -----------------------------
# 4. SCALE DATA
# -----------------------------
X_scaled = scaler.transform(X)

# -----------------------------
# 5. PREDICT
# -----------------------------
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

# -----------------------------
# 6. EVALUATION METRICS
# -----------------------------
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
roc = roc_auc_score(y, y_prob)

print("\n========= MODEL EVALUATION =========")
print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1-score  : {f1*100:.2f}%")
print(f"ROC AUC   : {roc:.4f}")

# -----------------------------
# 7. CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y, y_pred)

print("\n========= CONFUSION MATRIX =========")
print("              Predicted")
print("            Normal   Fall")
print(f"Actual Normal   {cm[0][0]}     {cm[0][1]}")
print(f"Actual Fall     {cm[1][0]}     {cm[1][1]}")

# OR display matrix nicely as table
cm_df = pd.DataFrame(
    cm,
    index=["Actual Normal", "Actual Fall"],
    columns=["Predicted Normal", "Predicted Fall"]
)

print("\nConfusion Matrix Table:")
print(cm_df)

# -----------------------------
# 8. CLASSIFICATION REPORT
# -----------------------------
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=["Normal", "Fall"]))
