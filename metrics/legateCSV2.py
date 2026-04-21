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

warnings.filterwarnings("ignore")

#Constants
root_dir = pathlib.Path('../anomaly_data/')
file_pattern_integrity = "anomaly_datasets*/integrity_dataset*.csv"
file_pattern_availability = "anomaly_datasets*/availability_dataset*.csv"

output_dir = pathlib.Path("../outputs/")
output_dir.mkdir(exist_ok=True)

model_file = "../models/og_random_forest.pkl"

def predict_anomalies(pattern: str):
    for file_path in root_dir.glob(pattern):
        print(f"Reading file: {file_path.name} from {file_path.parent}")

        dataframe = pd.read_csv(file_path, low_memory=False)
        dataframe.dropna(inplace=True)

        X_predict, encoders = prepare_features(dataframe)
        model = deserialize_model()['model']

        predictions = model.predict(X_predict)
        dataframe['anomaly_prediction'] = predictions

        anomalies = dataframe[dataframe['anomaly_prediction'] == 1]

        # 📁 output files
        report_file = output_dir / f"{file_path.stem}_report.txt"
        anomaly_file = output_dir / f"{file_path.stem}_anomalies.csv"

        with open(report_file, "w") as f:
            f.write(f"File: {file_path.name}\n")
            f.write(f"Total rows: {len(dataframe)}\n")
            f.write(f"Anomalies found: {len(anomalies)}\n\n")

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
    """
    Prepares the features from the input dataframe for prediction. This function replicates that
    of the process used during training, ensuring that the same preprocessing steps are applied.

    Args:
        dataframe (pd.DataFrame): The input dataframe containing raw data for prediction.
            Expected columns may include 'class', 'label', and 'device_label', which will
            be excluded from feature preparation.

    Returns:
        Tuple[pd.DataFrame, Dict[str, LabelEncoder]]: A tuple containing:
            - A dataframe with features preprocessed and ready for prediction.
            - A dictionary of encoders used during the preprocessing phase.
    """

    # Load the saved model and the encoders used during training
    encoders = deserialize_model().get('encoders', {})

    X_predict = dataframe.drop(['label', 'device_label'], axis=1, errors='ignore').copy()

    # Encode categorical columns -- apply encoders to new data
    for col, le in encoders.items():
        if col in X_predict.columns:
            X_predict[col] = X_predict[col].astype(str)
            # Use transform (NOT fit_transform) to maintain consistency with training
            X_predict[col] = X_predict[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

    # Convert all columns to numeric and handle any remaining objects
    # keep only numeric columns (same as training)
    X_predict = X_predict.select_dtypes(include=['number'])

    # fill any missing values just in case
    X_predict = X_predict.fillna(0)

    return X_predict, encoders


#REMOVE AFTER TESTING
if __name__ == '__main__':
    predict_anomalies(file_pattern_integrity)