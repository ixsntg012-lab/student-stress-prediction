"""
Student Stress Prediction — Flask Web App
==========================================
Routes:
  GET  /          → input form
  POST /predict   → prediction + recommendations
"""

from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# ── Load model ────────────────────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model, le, scaler = pickle.load(f)

# ── Recommendations ───────────────────────────────────────────────────────
def get_recommendations(data, prediction):
    tips = []

    if data["sleep_hours"] < 6:
        tips.append("😴 Try to sleep at least 7–8 hours daily — poor sleep is a major stress trigger.")
    if data["screen_time"] > 6:
        tips.append("📵 Reduce screen time to under 4 hours — excessive screen use increases anxiety.")
    if data["physical_activity"] < 2:
        tips.append("🏃 Add at least 30 minutes of physical activity daily — it significantly reduces stress.")
    if data["social_interaction"] < 2:
        tips.append("🤝 Stay connected with friends and family — social support lowers stress levels.")
    if data["study_hours"] < 3:
        tips.append("📚 Try to study at least 3–4 hours daily with proper breaks (Pomodoro technique).")
    if data["attendance"] < 70:
        tips.append("🏫 Improve your attendance — missing classes increases academic pressure.")
    if data["assignments"] < 60:
        tips.append("📝 Keep up with assignments — pending work is a hidden stress source.")

    if prediction == "low":
        tips = ["✅ Great job! You're managing stress well. Keep maintaining your healthy habits!"]

    if not tips:
        tips = ["👍 You're doing okay! Small improvements in sleep and activity can help further."]

    return tips


# ── Risk factors ──────────────────────────────────────────────────────────
def get_risk_factors(data):
    """Return top 3 factors contributing to stress."""
    scores = {
        "Sleep Hours":        max(0, (6 - data["sleep_hours"]) * 15),
        "Screen Time":        max(0, (data["screen_time"] - 4) * 10),
        "Physical Activity":  max(0, (3 - data["physical_activity"]) * 12),
        "Study Hours":        max(0, (4 - data["study_hours"]) * 8),
        "Attendance":         max(0, (70 - data["attendance"]) * 0.5),
        "Social Interaction": max(0, (3 - data["social_interaction"]) * 10),
        "Assignment Score":   max(0, (60 - data["assignments"]) * 0.5),
    }
    sorted_factors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [f for f, s in sorted_factors if s > 0][:3]


# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "study_hours":        float(request.form["study"]),
            "sleep_hours":        float(request.form["sleep"]),
            "attendance":         float(request.form["attendance"]),
            "assignments":        float(request.form["assignments"]),
            "screen_time":        float(request.form["screen"]),
            "physical_activity":  float(request.form["activity"]),
            "exam_score":         float(request.form["exam"]),
            "social_interaction": float(request.form["social"]),
        }

        df      = pd.DataFrame([data])
        scaled  = scaler.transform(df)
        pred    = model.predict(scaled)
        probs   = model.predict_proba(scaled)[0]
        result  = le.inverse_transform(pred)[0]

        # Rename low → No Stress
        display = "No Stress" if result == "low" else result.capitalize() + " Stress"

        # Probabilities
        prob_dict = {}
        for i, label in enumerate(le.classes_):
            name = "No Stress" if label == "low" else label.capitalize() + " Stress"
            prob_dict[name] = round(probs[i] * 100, 1)

        tips         = get_recommendations(data, result)
        risk_factors = get_risk_factors(data)

        return render_template(
            "index.html",
            prediction   = display,
            probabilities= prob_dict,
            tips         = tips,
            risk_factors = risk_factors,
            stress_level = result,
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)