from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.preprocessing import IQRClipper


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models/mental_health_pipeline.joblib"
DATASET_PATH = PROJECT_ROOT / "data/raw/student_mental_health.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def student_input(column):
    values = dataset[column].dropna()
    label = labels[column]
    if not pd.api.types.is_numeric_dtype(values):
        choices = sorted(values.unique().tolist())
        return st.selectbox(label, choices, index=choices.index(values.mode()[0]), key=column)

    if pd.api.types.is_integer_dtype(values):
        minimum, maximum, default = int(values.min()), int(values.max()), int(values.median())
        step = 1
    else:
        decimals = 2 if column == "GPA" else 1
        minimum, maximum = float(values.min()), float(values.max())
        default = round(float(values.median()), decimals)
        step = 0.01 if column == "GPA" else 0.1

    widget = st.number_input if column in ["Age", "Year_of_Study", "GPA", "Attendance_Rate"] else st.slider
    return widget(label, min_value=minimum, max_value=maximum, value=default, step=step, key=column)


st.set_page_config(page_title="Student Mental Health Risk Prediction")
st.title("Student Mental Health Risk Prediction")
st.write("An educational Machine Learning project using a synthetic student dataset.")
st.caption(
    "This prediction is an educational demonstration based on synthetic data. "
    "It is not a medical diagnosis or clinical risk assessment."
)

if not MODEL_PATH.is_file():
    st.error("Model missing. Run the Day 4 notebook to save models/mental_health_pipeline.joblib.")
    st.stop()
if not DATASET_PATH.is_file():
    st.error("Dataset missing. Put student_mental_health.csv in data/raw/.")
    st.stop()

model = load_model()
dataset = pd.read_csv(DATASET_PATH)
labels = {
    "Age": "Age (years)",
    "Gender": "Gender",
    "Year_of_Study": "Year of study",
    "Sleep_Hours": "Sleep hours per night",
    "Study_Hours_Per_Week": "Study hours per week",
    "GPA": "GPA (0–4 scale)",
    "Social_Media_Hours": "Social media hours per day",
    "Screen_Time_Hours": "Screen time hours per day",
    "Physical_Activity": "Physical activity",
    "Financial_Stress": "Financial stress (dataset score)",
    "Family_Support": "Family support (dataset score)",
    "Attendance_Rate": "Attendance rate (%)",
    "Part_Time_Job": "Part-time job",
    "Diet_Quality": "Diet quality",
    "Previous_Mental_Health_History": "Previous mental health history",
}
# Highest permutation importance on the original Day 4 holdout; no causal claim.
main_features = [
    "Sleep_Hours", "Financial_Stress", "Previous_Mental_Health_History",
    "Family_Support", "Physical_Activity", "GPA",
]

with st.form("student_information"):
    st.subheader("Student information")
    st.caption("Review all 15 inputs. Starting values use dataset medians or most common categories.")
    inputs = {column: student_input(column) for column in main_features}
    with st.expander("Other information"):
        for column in model.feature_names_in_:
            if column not in main_features:
                inputs[column] = student_input(column)
    submitted = st.form_submit_button("Predict Risk Level")

if submitted:
    input_df = pd.DataFrame([inputs], columns=model.feature_names_in_)
    input_df["Gender"] = input_df["Gender"].replace("Non-binary", "Unknown")
    prediction = model.predict(input_df)[0]
    risk_label = {0: "Low", 1: "Medium", 2: "High"}[int(prediction)]
    st.success(f"Predicted risk level: {risk_label}")
