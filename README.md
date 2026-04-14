# AIoT Health Risk Prediction Dashboard

An AIoT-based healthcare system that predicts patient risk levels (High Risk / Low Risk) using machine learning and visualizes insights through an interactive Streamlit dashboard.

---

## Live App

[https://health-risk-app-6dswcldv92hvzdzb9vbjuq.streamlit.app/](https://health-risk-app-6dswcldv92hvzdzb9vbjuq.streamlit.app/)

---

## Dataset

**Dataset Used:** Human Vital Signs Dataset  
[https://www.kaggle.com/datasets/nasirayub2/human-vital-sign-dataset/data](https://www.kaggle.com/datasets/nasirayub2/human-vital-sign-dataset/data)

**Features:** Heart Rate, Respiratory Rate, Body Temperature, Oxygen Saturation, Systolic Blood Pressure, Diastolic Blood Pressure, Age, Gender, Weight (kg), Height (m), Derived_HRV, Derived_Pulse_Pressure, Derived_BMI, Derived_MAP, Risk Category

---

## Project Objective

- Monitor patient vital parameters
- Predict health risk using machine learning
- Provide real-time visualization via an interactive dashboard

---

## Machine Learning Model

```python
DecisionTreeClassifier(max_depth=5, min_samples_split=10)
```

| Metric | Score |
|---|---|
| Train Accuracy | 91.55% |
| Test Accuracy | 91.34% |

---

## Features

- Interactive Streamlit dashboard
- Visualizations: Histogram, Scatter plot, Risk distribution, Correlation heatmap
- Real-time binary classification: High Risk / Low Risk

---

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit

---

## Workflow

Data Preprocessing → Feature Selection → Model Training → Evaluation → Streamlit Deployment

---

## Conclusion
The system predicts patient risk levels with over 91% accuracy, providing a reliable solution for smart healthcare monitoring using AIoT principles.

The system predicts patient risk levels with over **91% accuracy**, providing a reliable solution for smart healthcare monitoring using AIoT principles.
