"""
Student Stress Prediction — Model Training
==========================================
Trains and compares 3 ML models:
  - Logistic Regression
  - Random Forest
  - XGBoost

Auto-selects best model, saves:
  - model.pkl
  - confusion_matrix.png
  - feature_importance.png
  - eval_report.txt
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

os.makedirs("models", exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────────
df = pd.read_csv("dataset.csv")
print(f"Dataset size: {len(df)} samples")
print(f"Label distribution:\n{df['result'].value_counts()}\n")

# ── Encode labels ─────────────────────────────────────────────────────────
le = LabelEncoder()
df["result"] = le.fit_transform(df["result"])
print(f"Classes: {list(le.classes_)}\n")

# ── Features & target ─────────────────────────────────────────────────────
X = df.drop("result", axis=1)
y = df["result"]

# ── Scale ─────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Stratified train/test split ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

# ── Train 3 models ────────────────────────────────────────────────────────
print("Training models...")

lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))
print(f"Logistic Regression : {lr_acc*100:.2f}%")

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest       : {rf_acc*100:.2f}%")

xgb = XGBClassifier(eval_metric="mlogloss", random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
print(f"XGBoost             : {xgb_acc*100:.2f}%")

# ── Select best model ─────────────────────────────────────────────────────
models = {
    "Logistic Regression": (lr,  lr_acc),
    "Random Forest":       (rf,  rf_acc),
    "XGBoost":             (xgb, xgb_acc),
}
best_name  = max(models, key=lambda k: models[k][1])
best_model = models[best_name][0]
print(f"\nBest Model: {best_name} ({models[best_name][1]*100:.2f}%)")

# ── Cross validation ──────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv  = cross_val_score(best_model, X_scaled, y, cv=skf, scoring="accuracy")
print(f"5-Fold CV: {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")

# ── Final evaluation ──────────────────────────────────────────────────────
y_pred  = best_model.predict(X_test)
report  = classification_report(y_test, y_pred, target_names=le.classes_)
print(f"\nClassification Report:\n{report}")

# ── Save eval report ──────────────────────────────────────────────────────
with open("eval_report.txt", "w") as f:
    f.write("=== Student Stress Prediction — Model Evaluation ===\n\n")
    f.write(f"Dataset size  : {len(df)} samples\n")
    f.write(f"Train / Test  : {len(X_train)} / {len(X_test)}\n")
    f.write(f"Best Model    : {best_name}\n")
    f.write(f"Test Accuracy : {models[best_name][1]*100:.2f}%\n")
    f.write(f"5-Fold CV     : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%\n\n")
    f.write("Model Comparison:\n")
    for name, (_, acc) in models.items():
        f.write(f"  {name}: {acc*100:.2f}%\n")
    f.write(f"\nPer-Class Report:\n{report}")
print("[Saved] eval_report.txt")

# ── Save model ────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump((best_model, le, scaler), f)
print("[Saved] model.pkl")

# ── Confusion matrix ──────────────────────────────────────────────────────
cm   = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.close()
print("[Saved] confusion_matrix.png")

# ── Feature importance ────────────────────────────────────────────────────
if hasattr(best_model, "feature_importances_"):
    imp  = best_model.feature_importances_
    cols = list(X.columns)
    sorted_idx = np.argsort(imp)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([cols[i] for i in sorted_idx],
                  [imp[i] for i in sorted_idx],
                  color="#4C72B0", edgecolor="white")
    ax.set_title("Feature Importance", fontsize=13)
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance Score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.close()
    print("[Saved] feature_importance.png")

print(f"\n✓ Done! Best model: {best_name} — Accuracy: {models[best_name][1]*100:.2f}%")