import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# 1. Load dataset
df = pd.read_csv("dataset.csv")

# 2. Encode labels
le = LabelEncoder()
df["result"] = le.fit_transform(df["result"])

# 3. Features & target
X = df.drop("result", axis=1)
y = df["result"]

# 4. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL 1: Logistic Regression
# -------------------------------
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

print("\nLogistic Regression Accuracy:", lr_acc)

# -------------------------------
# MODEL 2: Random Forest
# -------------------------------
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_acc)

# -------------------------------
# MODEL 3: XGBoost
# -------------------------------
xgb_model = XGBClassifier(eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

print("XGBoost Accuracy:", xgb_acc)

# -------------------------------
# SELECT BEST MODEL
# -------------------------------
models = {
    "Logistic Regression": (lr_model, lr_acc),
    "Random Forest": (rf_model, rf_acc),
    "XGBoost": (xgb_model, xgb_acc)
}

best_name = max(models, key=lambda k: models[k][1])
best_model = models[best_name][0]

print("\nBest Model:", best_name)

# -------------------------------
# FINAL EVALUATION
# -------------------------------
y_pred = best_model.predict(X_test)

print("\nFinal Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# SAVE MODEL + SCALER
# -------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump((best_model, le, scaler), f)

# -------------------------------
# CROSS VALIDATION
# -------------------------------

print("\nCross Validation Scores:")

# Logistic Regression CV
lr_cv_scores = cross_val_score(LogisticRegression(max_iter=200), X_scaled, y, cv=5)
print("Logistic Regression CV Accuracy:", lr_cv_scores.mean())

# Random Forest CV
rf_cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100), X_scaled, y, cv=5)
print("Random Forest CV Accuracy:", rf_cv_scores.mean())

# -------------------------------
# FEATURE IMPORTANCE (RF only)
# -------------------------------
if best_name == "Random Forest":
    importances = best_model.feature_importances_

    plt.figure()
    plt.bar(X.columns, importances)
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.savefig("feature_importance.png")
    plt.show()

# -------------------------------
# CONFUSION MATRIX
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot()

plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()