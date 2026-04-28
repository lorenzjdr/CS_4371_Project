"""
CS_4371_Project — CSV Dataset Reader Utilities

Purpose
-------
Centralized helpers to load the project's CSV datasets. Import
functions from this module instead of calling `pandas.read_csv` directly so
that file paths, NA handling, encodings, and date parsing stay consistent
across the codebase.

Intended usage (examples)
-------------------------
from metrics.legateCSV import read_dataset  # to be implemented below

# Example: load labeled environment monitoring data
df = read_dataset(
    "Dataset/environmentMonitoring_labeled.csv",
    parse_dates=["timestamp"],
)

# Example: iterate over a large anomaly log in chunks
for chunk in read_dataset(
    "anomaly_data/anomaly_datasets5/anomaly_log5.csv", chunksize=100_000
):
    process(chunk)
"""

#Imports
import pathlib
import joblib
import pandas as pd
import numpy as np
import warnings

import sklearn.preprocessing
import sklearn.ensemble

warnings.filterwarnings("ignore")

#Constants
root_dir = pathlib.Path('../anomaly_data/')
file_pattern_integrity = "anomaly_datasets*/integrity_dataset*.csv"
file_pattern_availability = "anomaly_datasets*/availability_dataset*.csv"

output_dir = pathlib.Path("../outputs/")
output_dir.mkdir(exist_ok=True)

model_file = "../models/isolation_forest_model_environment.pkl"

def predict_anomalies(pattern: str):
    for file_path in root_dir.glob(pattern):
        print(f"Reading file: {file_path.name} from {file_path.parent}")

        dataframe = pd.read_csv(file_path, low_memory=False)
        dataframe.dropna(inplace=True)

        X_predict, encoders = prepare_features(dataframe)
        model = deserialize_model()['model']

        X_predict = X_predict.reindex(columns=model.feature_names_in_, fill_value=0)
        predictions = model.predict(X_predict)
        dataframe['anomaly_prediction'] = predictions

        anomaly_label = -1 if isinstance(model, sklearn.ensemble.IsolationForest) else 1
        normal_label = 1 if anomaly_label == -1 else 0

        anomalies = dataframe[dataframe['anomaly_prediction'] == anomaly_label]
        anomalies_true = len(dataframe[dataframe['label'] == 1])
        anomalies_false = len(dataframe[dataframe['label'] == 0])

        # output files
        report_file = output_dir / f"{file_path.stem}_report.txt"
        anomaly_file = output_dir / f"{file_path.stem}_anomalies.csv"

        #ticker
        anomaly_count = len(dataframe)
        true_positive = len(dataframe[(dataframe['anomaly_prediction'] == anomaly_label) & (dataframe['label'] == 1)]) + 0.0001
        false_positive = len(dataframe[(dataframe['anomaly_prediction'] == anomaly_label) & (dataframe['label'] == 0)])
        true_negative = len(dataframe[(dataframe['anomaly_prediction'] == normal_label) & (dataframe['label'] == 0)])
        false_negative = len(dataframe[(dataframe['anomaly_prediction'] == normal_label) & (dataframe['label'] == 1)])

        with open(report_file, "w") as f:
            f.write(f"File: {file_path.name}\n")
            f.write(f"Total rows: {len(dataframe)}\n")
            f.write(f"Anomalies found: {len(anomalies)}\n\n")
            f.write(f"Anomalies total: {anomalies_true}\n")

            f.write(f"Accuracy: {round((true_positive + true_negative) / anomaly_count * 100, 3)}%\n")#TP + TN / (TP + TN + FP + FN) <- total
            f.write(f"Precision: {round(true_positive / (true_positive + false_positive), 3)}\n")#TP / (TP + FP)
            f.write(f"Recall: {round(true_positive / (true_positive + false_negative), 3)}\n\n")#TP / (TP + FN)

            if not anomalies.empty:
                f.write("ANOMALY ROW INDICES:\n")
                for idx in anomalies.index:
                    f.write(f"{idx}\n")

                anomalies.to_csv(anomaly_file, index=False)

                f.write(f"\nSaved anomaly CSV: {anomaly_file.name}\n")
            else:
                f.write("No anomalies detected.\n")


def deserialize_model():
    return joblib.load(model_file) #hardcoded model file path

def prepare_features(dataframe : pd.DataFrame):
    """Encode categorical features and return X"""
    # Drop label columns and non-feature columns
    X = dataframe.drop(['class', 'label', 'device_label'], axis=1, errors='ignore').copy()

    # Encode categorical columns
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        le = sklearn.preprocessing.LabelEncoder()
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Convert all columns to numeric and handle any remaining objects
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    # Ensure all columns are numeric
    X = X.astype(float)

    print(f"Features shape: {X.shape}")
    return X, label_encoders


#REMOVE AFTER TESTING
if __name__ == '__main__':
    predict_anomalies(file_pattern_integrity)