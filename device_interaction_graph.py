import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
 
DEFAULT_DATASET = "anomaly_datasets5/integrity_dataset5.csv"
 
FEATURE_COLS = [
    "frame.len",
    "tcp.window_size_value",
    "tcp.len",
    "tcp.ack",
    "frame.time_delta",
    "tcp.dstport",
    "mqtt.msgtype",
    "mqtt.qos",
    "ip.ttl",
    "tcp.flags.reset",
    "tcp.flags.syn",
    "tcp.flags.push",
    "tcp.flags.ack",
]
 
IF_CONTAMINATION = 0.0015
 
def load_data(path):
    print(f"Loading: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Rows: {len(df):,}  |  Anomalies (label=1): {(df['label'] == 1).sum()}")
 
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  Warning: missing columns (will be skipped): {missing}")
 
    X = df[available].fillna(0).astype(float)
    y = df["label"].astype(int)
    return X, y, available

def plot_confusion_matrix(y_test, y_pred, output_path):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
 
    fig, ax = plt.subplots(figsize=(5, 4))
 
    cmap = sns.color_palette("Blues", as_cmap=True)
 
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        linewidths=0.5,
        linecolor="#dddddd",
        annot_kws={"size": 13, "weight": "bold", "color": "white"},
        ax=ax,
    )
 
    for text, val in zip(ax.texts, cm.flatten()):
        text.set_color("white" if val > cm.max() * 0.3 else "#333333")
 
    ax.set_title("Random Forest — Confusion Matrix", fontsize=12,
                 fontweight="bold", pad=10)
    ax.set_xlabel("Predicted", fontsize=10, labelpad=8)
    ax.set_ylabel("Actual", fontsize=10, labelpad=8)
    ax.tick_params(axis="both", labelsize=9)
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

def plot_feature_importances(rf_model, feature_names, output_path):
    importances = pd.Series(rf_model.feature_importances_,
                            index=feature_names).sort_values(ascending=True)
 
    norm = importances / importances.max()
    colors = [plt.cm.Blues(0.35 + 0.65 * v) for v in norm]
 
    fig, ax = plt.subplots(figsize=(8, 5))
 
    bars = ax.barh(importances.index, importances.values,
                   color=colors, edgecolor="none", height=0.65)
 
    ax.set_title("Feature Importances (Random Forest)", fontsize=12,
                 fontweight="bold", pad=10)
    ax.set_xlabel("Importance", fontsize=10)
    ax.set_facecolor("#f8eded")
    ax.grid(axis="x", color="white", linewidth=1.3, zorder=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xlim(0, importances.max() * 1.12)
 
    for bar, val in zip(bars, importances.values):
        if val > 0.005:
            ax.text(val + importances.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7.5, color="#333333")
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

def plot_anomaly_scores(scores, preds, output_path):
    """
    scores : array of anomaly scores from iso.decision_function()
             positive = normal, negative = anomaly
    preds  : array of predictions (1 = normal, -1 = anomaly)
    """
    packet_idx = np.arange(len(scores))
    is_anomaly = preds == -1
 
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_facecolor("#ebebeb")
 
    ax.fill_between(packet_idx, scores, 0,
                    where=~is_anomaly,
                    color="#3d3d99", alpha=0.65, linewidth=0,
                    label="_nolegend_")
 
    ax.fill_between(packet_idx, scores, 0,
                    where=is_anomaly,
                    color="#cc2222", alpha=0.85, linewidth=0,
                    label="_nolegend_")
 
    ax.axhline(0, color="#cc2222", linewidth=1.5, linestyle="--", zorder=3)
 
    ax.set_title("Isolation Forest — Anomaly Scores Over Time",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Packet index", fontsize=10)
    ax.set_ylabel("Anomaly score\n(< 0 = anomaly)", fontsize=9)
    ax.grid(axis="y", color="white", linewidth=1.0, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
 
    legend_handles = [
        mpatches.Patch(color="#cc2222", linestyle="--",
                       label=f"Anomaly threshold  (detected: {is_anomaly.sum()})"),
        mpatches.Patch(color="#cc2222", alpha=0.85, label="Anomaly"),
    ]
    ax.legend(handles=legend_handles, loc="lower left",
              fontsize=8.5, framealpha=0.85, facecolor="white",
              edgecolor="#cccccc")
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate anomaly detection charts")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Path to anomaly dataset CSV")
    parser.add_argument("--output", default="chart_output",
                        help="Output directory for chart PNGs")
    args = parser.parse_args()
 
    os.makedirs(args.output, exist_ok=True)
 
    # ── Load data
    X, y, features = load_data(args.dataset)
 
    # ── Random Forest
    print("\n[1/3] Training Random Forest...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
        stratify=y if y.sum() > 1 else None
    )
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
 
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
 
    plot_confusion_matrix(
        y_test, y_pred,
        os.path.join(args.output, "rf_confusion_matrix.png")
    )
    plot_feature_importances(
        rf, features,
        os.path.join(args.output, "rf_feature_importances.png")
    )
 
    # ── Isolation Forest
    print("\n[2/3] Training Isolation Forest...")
    iso = IsolationForest(contamination=IF_CONTAMINATION, random_state=42)
    iso.fit(X)
    scores = iso.decision_function(X)
    preds  = iso.predict(X)
    print(f"  Anomalies detected: {(preds == -1).sum()}")
 
    plot_anomaly_scores(
        scores, preds,
        os.path.join(args.output, "if_anomaly_scores.png")
    )
 
    print(f"\nDone. Charts saved to: {args.output}/")
 
 
if __name__ == "__main__":
    main()