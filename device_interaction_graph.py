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
 
# ── Config ─────────────────────────────────────────────────────────────────
 
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
 
# Isolation Forest contamination — roughly the expected anomaly rate
IF_CONTAMINATION = 0.0015
 
 
# ── Helpers ────────────────────────────────────────────────────────────────
 
def load_data(path):
    print(f"Loading: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Rows: {len(df):,}  |  Anomalies (label=1): {(df['label'] == 1).sum()}")
 
    # Keep only features that exist in this CSV
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  Warning: missing columns (will be skipped): {missing}")
 
    X = df[available].fillna(0).astype(float)
    y = df["label"].astype(int)
    return X, y, available