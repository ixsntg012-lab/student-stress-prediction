# Student Stress Predictor

A machine learning web application that predicts student stress levels based on academic and lifestyle habits — and provides **personalized recommendations** to help reduce stress.

![Demo](screenshot.png)

---

## Problem Statement

Student mental health is a growing concern in universities worldwide. High stress levels lead to poor academic performance, burnout, and mental health issues. This system helps students identify their stress level early and take corrective action — based on their own daily habits.

---

## What It Does

1. Student enters 8 lifestyle factors (sleep, study, screen time, etc.)
2. ML model predicts stress level: **High / Medium / No Stress**
3. Confidence scores shown for each class
4. **Primary risk factors** highlighted (e.g. "Screen Time", "Sleep Hours")
5. **Personalized recommendations** given based on specific inputs

---

## Features

| Feature | Description |
|---|---|
| Multi-model comparison | Trains Logistic Regression, Random Forest, XGBoost — auto-selects best |
| Confidence scores | Probability for each stress class shown as progress bars |
| Risk factor identification | Top 3 contributing factors highlighted per prediction |
| Personalized recommendations | Specific tips based on user's actual input values |
| Clean web UI | Flask + responsive HTML/CSS — works on mobile and desktop |
| Model evaluation | Confusion matrix, feature importance, classification report saved |
| Cross-validation | 5-fold stratified CV for reliable accuracy estimate |

---

## Input Features

| Feature | Range | Description |
|---|---|---|
| Study Hours | 0–12 | Daily study hours |
| Sleep Hours | 0–12 | Nightly sleep hours |
| Attendance | 0–100 | Class attendance percentage |
| Assignment Score | 0–100 | Average assignment score |
| Screen Time | 0–16 | Daily screen time (hours) |
| Physical Activity | 0–6 | Days per week exercising |
| Exam Score | 0–100 | Latest exam score |
| Social Interaction | 0–6 | Days per week socializing |

---

## ML Pipeline

```
Dataset (1,200 samples)
     │
     ▼
StandardScaler (feature normalization)
     │
     ▼
Train 3 Models:
  ├── Logistic Regression
  ├── Random Forest
  └── XGBoost
     │
     ▼
Auto-select best model (highest test accuracy)
     │
     ▼
5-Fold Stratified Cross Validation
     │
     ▼
Save model.pkl + eval_report.txt + confusion_matrix.png
     │
     ▼
Flask Web App → Prediction + Risk Factors + Recommendations
```

---

## Dataset

- **Size:** 1,200 samples (400 per class)
- **Generated:** Programmatically using realistic behavioral patterns
- **Classes:** High Stress, Medium Stress, No Stress (balanced)
- **Patterns used:**
  - High stress → low sleep, low study, high screen time, low activity
  - Medium stress → moderate values across all features
  - No stress → good sleep, regular study, low screen, high activity

---

## Tech Stack

| Component | Technology |
|---|---|
| ML Models | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Web Framework | Flask |
| Frontend | HTML, CSS (no external libraries) |
| Visualization | Matplotlib |
| Model Storage | Pickle |

---

## Installation

```bash
git clone https://github.com/ixsntg012-lab/student-stress-prediction.git
cd student-stress-prediction
pip install -r requirements.txt
```

---

## Usage

```bash
# Step 1 — Generate dataset
python generate_dataset.py

# Step 2 — Train model
python train.py

# Step 3 — Run web app
python app.py
```

Open browser: `http://localhost:5000`

---

## Project Structure

```
student-stress-prediction/
│
├── templates/
│   └── index.html           ← Web UI
│
├── generate_dataset.py      ← Dataset generation
├── train.py                 ← Model training + evaluation
├── app.py                   ← Flask web application
├── predict.py               ← CLI prediction tool
├── dataset.csv              ← Training data
├── model.pkl                ← Trained model (generated)
├── confusion_matrix.png     ← Model evaluation (generated)
├── feature_importance.png   ← Feature analysis (generated)
├── eval_report.txt          ← Accuracy report (generated)
├── requirements.txt
└── README.md
```

---

## Results

See `eval_report.txt` after running `train.py` for:
- Test accuracy per model
- 5-fold CV accuracy
- Per-class precision, recall, F1

---

## Key Insights

- **Screen time** and **sleep hours** are the strongest stress predictors
- **Physical activity** and **social interaction** are protective factors
- Balanced lifestyle (study + sleep + exercise) consistently predicts low stress

---

## Limitations & Future Work

- Dataset is synthetically generated — real-world survey data would improve generalization
- Future: integrate with university systems for real student data
- Future: add longitudinal tracking (stress trend over weeks)
- Future: deploy on cloud (Heroku / Render / Hugging Face)

---

## Author

**Swetha Kiran Veernapu**
MS Computer Science

---

## License

MIT License