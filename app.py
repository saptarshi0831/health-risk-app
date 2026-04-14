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

# load data
@st.cache_data
def load_data():
    data = pd.read_csv("human_vital_signs_dataset_2024_small.csv")

    data = data.drop(columns=['Patient ID', 'Timestamp'], errors='ignore')

    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

    if 'Risk Category' in data.columns:
        data['Risk Category'] = data['Risk Category'].map({
            'Low Risk': 0,
            'High Risk': 1
        })

    return data

data = load_data()

# feature selection
drop_cols = [
    'Risk Category',
    'Derived_BMI',
    'Derived_MAP',
    'Derived_Pulse_Pressure',
    'Derived_HRV'
]

drop_cols = [col for col in drop_cols if col in data.columns]

X = data.drop(columns=drop_cols)
y = data['Risk Category']

# model train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
model.fit(X_train, y_train)

# metrics card
col1, col2, col3 = st.columns(3)

col1.metric("Model Accuracy", f"{round(model.score(X_test, y_test)*100,2)}%")
col2.metric("Total Patients", len(data))
col3.metric("Features Used", X.shape[1])

# graphs
st.subheader("Data Insights")

col1, col2 = st.columns(2)

with col1:
    if 'Heart Rate' in data.columns:
        fig, ax = plt.subplots()
        data['Heart Rate'].hist(ax=ax)
        ax.set_title("Heart Rate Distribution")
        st.pyplot(fig)

with col2:
    if 'Risk Category' in data.columns:
        fig, ax = plt.subplots()
        sns.countplot(x='Risk Category', data=data, ax=ax)
        ax.set_title("Risk Distribution")
        st.pyplot(fig)
        
# input sidebar
st.sidebar.header("Patient Input")

def get_input(feature, default, min_val, max_val):
    if feature in X.columns:
        return st.sidebar.slider(feature, min_val, max_val, default)
    else:
        return default

input_dict = {}

for col in X.columns:
    if col == 'Gender':
        gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
        input_dict[col] = 1 if gender == "Male" else 0

    elif col == 'Heart Rate':
        input_dict[col] = get_input(col, 80, 50, 150)

    elif col == 'Body Temperature':
        input_dict[col] = get_input(col, 37.0, 35.0, 42.0)

    elif col == 'Oxygen Saturation':
        input_dict[col] = get_input(col, 98, 80, 100)

    elif col == 'Age':
        input_dict[col] = get_input(col, 30, 1, 100)

    else:
        input_dict[col] = st.sidebar.number_input(col, value=0.0)

input_data = pd.DataFrame([input_dict])

# prediction
st.subheader("Prediction")

if st.button("Predict Risk"):

    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("LOW RISK - Patient is Stable")
    else:
        st.error("HIGH RISK - Immediate Attention Needed!")
