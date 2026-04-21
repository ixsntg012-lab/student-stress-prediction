"""
Generate realistic student stress dataset — 1200 samples
Patterns:
  High stress   → low sleep, low study, high screen, low activity
  Medium stress → moderate everything
  Low stress    → good sleep, good study, low screen, high activity
"""
import numpy as np
import pandas as pd

np.random.seed(42)

def generate(n, stress_level):
    if stress_level == "high":
        study      = np.random.randint(1, 4,   n)
        sleep      = np.random.randint(3, 6,   n)
        attendance = np.random.randint(40, 65, n)
        assignment = np.random.randint(30, 55, n)
        screen     = np.random.randint(7, 12,  n)
        activity   = np.random.randint(0, 2,   n)
        exam       = np.random.randint(30, 55, n)
        social     = np.random.randint(0, 2,   n)

    elif stress_level == "medium":
        study      = np.random.randint(3, 7,   n)
        sleep      = np.random.randint(5, 8,   n)
        attendance = np.random.randint(60, 80, n)
        assignment = np.random.randint(50, 75, n)
        screen     = np.random.randint(4, 8,   n)
        activity   = np.random.randint(2, 4,   n)
        exam       = np.random.randint(50, 75, n)
        social     = np.random.randint(2, 4,   n)

    else:  # low
        study      = np.random.randint(6, 12,  n)
        sleep      = np.random.randint(7, 10,  n)
        attendance = np.random.randint(80, 100,n)
        assignment = np.random.randint(75, 100,n)
        screen     = np.random.randint(1, 5,   n)
        activity   = np.random.randint(4, 7,   n)
        exam       = np.random.randint(75, 100,n)
        social     = np.random.randint(4, 7,   n)

    return pd.DataFrame({
        "study_hours":        study,
        "sleep_hours":        sleep,
        "attendance":         attendance,
        "assignments":        assignment,
        "screen_time":        screen,
        "physical_activity":  activity,
        "exam_score":         exam,
        "social_interaction": social,
        "result":             stress_level
    })

high   = generate(400, "high")
medium = generate(400, "medium")
low    = generate(400, "low")

df = pd.concat([high, medium, low]).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("dataset.csv", index=False)

print(f"Total samples: {len(df)}")
print(df["result"].value_counts())
print("\nSample high stress:")
print(df[df["result"]=="high"].head(3).to_string())
print("\nSample low stress:")
print(df[df["result"]=="low"].head(3).to_string())