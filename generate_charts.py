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

# ── Chart 2 — Feature Importances ────────────────────────────────────────────
def plot_feature_importances(rf_model, feature_names: list, out_path: str, top_n: int = 10):
    importances = rf_model.feature_importances_
    idx         = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in idx]
    top_values   = importances[idx]
 
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.85, top_n))
    bars = ax.barh(top_features, top_values, color=colors, edgecolor="white", linewidth=0.7)
 
    for bar, val in zip(bars, top_values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8.5, color="#333333")
 
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(f"Our Random Forest — Top {top_n} Feature Importances",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlim(0, top_values.max() * 1.18)
 
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    X, y, og_preds, features = load_data()
    print(f"  Total samples: {len(y)}  |  Normal: {(y==0).sum()}  |  Anomaly: {(y==1).sum()}")
 
    X_train, X_test, y_train, y_test, og_tr, og_te = train_test_split(
        X, y, og_preds, test_size=0.3, random_state=42, stratify=y
    )
 
    print("\n[1/2] Training Our Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
 
    print("[2/2] Training Isolation Forest...")
    iso = IsolationForest(contamination=0.4, random_state=42)
    iso.fit(X_train)
    iso_preds = np.where(iso.predict(X_test) == -1, 1, 0)
 
    metrics = {
        "OG Random Forest":  compute_metrics(y_test, og_te),
        "Our Random Forest": compute_metrics(y_test, rf_preds),
        "Isolation Forest":  compute_metrics(y_test, iso_preds),
    }
 
    print("\n  Metrics summary:")
    for model, vals in metrics.items():
        print(f"    {model:22s}  acc={vals[0]:.3f}  prec={vals[1]:.3f}  rec={vals[2]:.3f}  f1={vals[3]:.3f}")
 
    print("\nGenerating charts...")
    plot_model_comparison(metrics, str(OUT_DIR / "1_model_comparison.png"))
    plot_feature_importances(rf, features, str(OUT_DIR / "2_rf_feature_importances.png"))
 
    print(f"\nDone. Charts saved to ./{OUT_DIR}/")
 
 
if __name__ == "__main__":
    main()