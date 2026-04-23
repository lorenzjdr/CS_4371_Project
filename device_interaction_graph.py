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