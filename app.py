import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="AIoT Health Dashboard", layout="wide")

# CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1 {
    color: #2c3e50;
}
.stButton>button {
    background-color: #3498db;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# title
st.title("AIoT Smart Healthcare Dashboard")

# data load
import os

@st.cache_data
def load_data():
    filename = "human_vital_signs_dataset_2024_small.csv"
    
    if not os.path.exists(filename):
        import subprocess
        subprocess.run([
            "wget",
            "https://raw.githubusercontent.com/saptarshi0831/health-risk-app/main/human_vital_signs_dataset_2024_small.csv"
        ])
    
    data = pd.read_csv(filename)

    data = data.drop(columns=['Patient ID', 'Timestamp'])
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data['Risk Category'] = data['Risk Category'].map({
        'Low Risk': 0,
        'High Risk': 1
    })

    return data

data = load_data()

# input sidebar
st.sidebar.header("Patient Input")

heart_rate = st.sidebar.slider("Heart Rate", 50, 150, 80)
temp = st.sidebar.slider("Temperature", 35.0, 42.0, 37.0)
spo2 = st.sidebar.slider("SpO₂", 80, 100, 98)
age = st.sidebar.slider("Age", 1, 100, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

gender = 1 if gender == "Male" else 0

# model
X = data.drop(columns=[
    'Risk Category',
    'Derived_BMI',
    'Derived_MAP',
    'Derived_Pulse_Pressure',
    'Derived_HRV'
])
y = data['Risk Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
model.fit(X_train, y_train)

# metrics card
col1, col2, col3 = st.columns(3)

col1.metric("Model Accuracy", f"{round(model.score(X_test,y_test)*100,2)}%")
col2.metric("Total Patients", len(data))
col3.metric("Features Used", X.shape[1])

# graphs
st.subheader("Data Insights")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    data['Heart Rate'].hist(ax=ax)
    ax.set_title("Heart Rate Distribution")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.countplot(x='Risk Category', data=data, ax=ax)
    ax.set_title("Risk Distribution")
    st.pyplot(fig)

# prediction
input_data = pd.DataFrame([{
    'Heart Rate': heart_rate,
    'Respiratory Rate': 20,
    'Body Temperature': temp,
    'Oxygen Saturation': spo2,
    'Systolic Blood Pressure': 120,
    'Diastolic Blood Pressure': 80,
    'Age': age,
    'Gender': gender,
    'Weight (kg)': 70,
    'Height (m)': 1.7
}])

st.subheader("Prediction")

if st.button("Predict Risk"):

    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("LOW RISK")
    else:
        st.error("HIGH RISK")
