Student Stress Prediction System

.....................................................

Overview

This project predicts student stress levels based on academic and lifestyle factors using machine learning. It analyzes behavioral patterns and provides real-time predictions along with confidence scores.

.....................................................

Features
Predicts stress level: Low / Medium / High
Uses multiple models: Logistic Regression, Random Forest, XGBoost
Model comparison and selection
Cross-validation for reliability
Feature importance analysis
Confusion matrix evaluation
Probability-based predictions

.....................................................

Input Features
Study Hours (0–12)
Sleep Hours (0–12)
Attendance (0–100)
Assignments Score (0–100)
Screen Time (hours)
Physical Activity Level (0–6)
Exam Score (0–100)
Social Interaction Level (0–6)

.....................................................

Tech Stack
Python
Pandas
Scikit-learn
XGBoost
Matplotlib

.....................................................

How It Works
Data preprocessing and scaling
Training multiple machine learning models
Cross-validation for performance evaluation
Selection of best-performing model
Prediction using user input

.....................................................

Output
Predicted stress level
Probability for each class

Example:
Predicted Stress Level: High
High: 78%
Medium: 16%
Low: 6%

.....................................................

How to Run

Install dependencies:
pip install -r requirements.txt

Train the model:
python train.py

Run prediction:
python predict.py

.....................................................

Results
High accuracy achieved on structured dataset
Logistic Regression performed best in this setup
Feature importance highlights key stress factors

.....................................................

Key Insights
High screen time and low sleep increase stress
Physical activity and social interaction reduce stress
Balanced lifestyle leads to lower stress levels

.....................................................

Future Improvements
Use larger real-world datasets
Deploy as a web application
Integrate intelligent recommendation system

.....................................................

Author

Swetha Kiran Veernapu