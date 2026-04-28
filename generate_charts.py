"""
generate_charts.py
==================
Generates 2 anomaly detection visualizations:
  1. Grouped bar chart — Model comparison (Accuracy, Precision, Recall, F1)
  2. Feature importance bar chart — Our Random Forest top 10 features
 
Usage:
    python generate_charts.py
 
Output:
    chart_output/1_model_comparison.png
    chart_output/2_rf_feature_importances.png
"""
 
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
 
# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path("../outputs")
OUT_DIR  = pathlib.Path("chart_output")
OUT_DIR.mkdir(exist_ok=True)
 
DATASETS = [
    "integrity_dataset3_anomalies.csv",
    "integrity_dataset5_anomalies.csv",
    "integrity_dataset50_anomalies.csv",
]
 
COLORS = {
    "OG Random Forest":  "#4C72B0",
    "Our Random Forest": "#55A868",
    "Isolation Forest":  "#C44E52",
}
 
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.facecolor":    "#f8f8f8",
    "figure.facecolor":  "white",
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame):
    drop_cols = ["label", "device_label", "ip.dst", "ip.src", "anomaly_prediction", "class"]
    X = df.drop(drop_cols, axis=1, errors="ignore").copy()
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    return X.astype(float)