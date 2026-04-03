import pickle
import pandas as pd

# load model
with open("model.pkl", "rb") as f:
    model, le, scaler = pickle.load(f)

def get_input():
    try:
        study_hours = float(input("Study hours: "))
        sleep_hours = float(input("Sleep hours: "))
        attendance = float(input("Attendance: "))
        assignments = float(input("Assignments: "))
        screen_time = float(input("Screen time: "))
        physical_activity = float(input("Physical activity: "))
        exam_score = float(input("Exam score: "))
        social_interaction = float(input("Social interaction: "))

        return [[study_hours, sleep_hours, attendance, assignments,
                 screen_time, physical_activity, exam_score, social_interaction]]

    except:
        print("Invalid input")
        return None

sample = get_input()

if sample:
    columns = [
        "study_hours", "sleep_hours", "attendance", "assignments",
        "screen_time", "physical_activity", "exam_score", "social_interaction"
    ]

    sample_df = pd.DataFrame(sample, columns=columns)

    # scale input
    sample_scaled = scaler.transform(sample_df)

    # prediction
    pred = model.predict(sample_scaled)
    probs = model.predict_proba(sample_scaled)[0]

    # labels
    labels = le.classes_

    print("\nPredicted Stress Level:", le.inverse_transform(pred)[0])

    print("\nPrediction Probabilities:")
    for label, prob in zip(labels, probs):

        print(f"{label}: {prob*100:.2f}%")