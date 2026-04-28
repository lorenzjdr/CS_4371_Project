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

def load_data():
    dfs = []
    for fname in DATASETS:
        path = DATA_DIR / fname
        if path.exists():
            dfs.append(pd.read_csv(path, low_memory=False))
        else:
            print(f"  [WARNING] {path} not found — skipping.")
    if not dfs:
        raise FileNotFoundError(f"No dataset files found in {DATA_DIR}")
    combined = pd.concat(dfs, ignore_index=True)
    X = prepare_features(combined)
    y = combined["label"].values
    og_preds_raw = combined["anomaly_prediction"].values
    og_preds = np.where(og_preds_raw == -1, 1, og_preds_raw)
    return X, y, og_preds, X.columns.tolist()

def compute_metrics(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
    ]

# ── Chart 1 — Model Comparison ────────────────────────────────────────────────
def plot_model_comparison(metrics: dict, out_path: str):
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    n_metrics = len(metric_names)
    n_models  = len(metrics)
    x       = np.arange(n_metrics)
    width   = 0.22
    offsets = np.linspace(-1, 1, n_models) * width
 
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (model, vals) in enumerate(metrics.items()):
        bars = ax.bar(x + offsets[i], vals, width, label=model,
                      color=COLORS[model], alpha=0.88, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#333333")
 
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Performance Comparison\n(OG Random Forest vs Our Random Forest vs Isolation Forest)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
 
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
