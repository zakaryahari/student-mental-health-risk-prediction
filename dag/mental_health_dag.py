from datetime import datetime, timezone
from pathlib import Path

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator


DATASET_PATH = Path("/workspace/data/raw/student_mental_health.csv")
MODEL_PATH = Path("/workspace/models/mental_health_pipeline.joblib")


def check_dataset():
    import pandas as pd

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"CSV missing: put student_mental_health.csv at {DATASET_PATH}")
    dataset = pd.read_csv(DATASET_PATH)
    print("Dataset shape:", dataset.shape)
    print("Dataset OK")


def check_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model missing: {MODEL_PATH}. Run the Day 4 notebook first.")
    print("Model found")


def test_prediction():
    import joblib
    import pandas as pd
    from src.preprocessing import IQRClipper

    model = joblib.load(MODEL_PATH)
    example = pd.read_csv(DATASET_PATH, nrows=1)
    example = example.drop(columns=["Student_ID", "Mental_Health_Risk"])
    example["Gender"] = example["Gender"].replace("Non-binary", "Unknown")
    prediction = model.predict(example)[0]
    label = {0: "Low", 1: "Medium", 2: "High"}[int(prediction)]
    print(f"Predicted class: {prediction} ({label})")
    print("Synthetic dataset demonstration only; not a clinical prediction service.")


with DAG(
    dag_id="student_mental_health_day5",
    schedule=None,
    start_date=datetime(2026, 11, 1, tzinfo=timezone.utc),
    catchup=False,
) as dag:
    check_dataset = PythonOperator(task_id="check_dataset", python_callable=check_dataset)
    check_model = PythonOperator(task_id="check_model", python_callable=check_model)
    test_prediction = PythonOperator(task_id="test_prediction", python_callable=test_prediction)

    check_dataset >> check_model >> test_prediction
