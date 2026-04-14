"""
Detection + Rollback Runner

Loads the trained Isolation Forest model, runs it against attack data,
and triggers the rollback mechanism for any anomalous devices detected.

Usage:
    python detect_and_rollback.py
"""

import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from rollback import NetworkGraph, RollbackManager
from label_data import detect_gateway


MODEL_PATH = "models/isolation_forest_model_environment.pkl"
DEVICE_MAPPING_PATH = "device_mapping.pkl"
ATTACK_DATA_PATH = "Dataset/Attack_labeled.csv"
ENV_DATA_PATH = "Dataset/environmentMonitoring.csv"


def load_model(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["encoders"]


def load_device_mapping(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["device_mapping"]


def prepare_features(df, model_features):
    """
    Encode categorical columns and align to the feature set the model was trained on.
    Mirrors the logic in train_model.py's prepare_features().
    """
    X = df.drop(["class", "label", "device_label"], axis=1, errors="ignore").copy()

    # Encode categorical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])

    # Convert any remaining object columns to numeric
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    X = X.astype(float)

    # Align columns to what the model expects — fill missing with 0, drop extras
    X = X.reindex(columns=model_features, fill_value=0)

    return X


def main():
    print("=" * 65)
    print("IoT-AD: Detection + Rollback")
    print("=" * 65)

    # --- Load model ---
    print(f"\nLoading model: {MODEL_PATH}")
    model, _ = load_model(MODEL_PATH)
    print(f"Model expects {len(model.feature_names_in_)} features")

    # --- Load device mapping ---
    print(f"Loading device mapping: {DEVICE_MAPPING_PATH}")
    device_mapping = load_device_mapping(DEVICE_MAPPING_PATH)
    print(f"Loaded {len(device_mapping)} devices")

    # --- Detect gateway from environment data ---
    print(f"Detecting gateway from: {ENV_DATA_PATH}")
    env_df = pd.read_csv(ENV_DATA_PATH, low_memory=False)
    gateway_ip = detect_gateway(env_df)

    # --- Build network graph ---
    net_graph = NetworkGraph()
    net_graph.build_from_device_mapping(device_mapping, gateway_ip)
    print(f"Network graph: {net_graph.graph.number_of_nodes()} nodes, "
          f"{net_graph.graph.number_of_edges()} edges")

    rollback = RollbackManager(net_graph)

    # --- Load attack data ---
    print(f"\nLoading attack data: {ATTACK_DATA_PATH}")
    attack_df = pd.read_csv(ATTACK_DATA_PATH, low_memory=False)
    attack_df = attack_df.dropna(subset=["ip.src"])
    print(f"Loaded {len(attack_df)} attack records")

    # --- Prepare features and run detection ---
    print("\nRunning anomaly detection...")
    X = prepare_features(attack_df, model.feature_names_in_)

    scores = model.decision_function(X)   # lower = more anomalous
    preds = model.predict(X)              # -1 = anomaly, 1 = normal

    attack_df = attack_df.reset_index(drop=True)
    anomalous_mask = preds == -1
    anomalous_count = anomalous_mask.sum()
    print(f"Detected {anomalous_count} anomalous packets out of {len(preds)} total")

    # --- Trigger rollback for each unique anomalous device ---
    print("\nTriggering rollback for anomalous devices...")
    triggered = set()
    anomalous_rows = attack_df[anomalous_mask]

    for idx, row in anomalous_rows.iterrows():
        ip = row.get("ip.src")
        if pd.isna(ip) or ip in triggered:
            continue

        # If the IP is not a known device, register it as an attacker node
        # connected to the gateway so it can still be isolated
        if ip not in net_graph.graph.nodes:
            label = row.get("device_label")
            label = label if pd.notna(label) else f"Attacker ({ip})"
            net_graph.add_device(ip, label=label)
            net_graph.add_comm_path(ip, gateway_ip)
            net_graph.add_comm_path(gateway_ip, ip)

        score = float(scores[idx])
        rollback.trigger(ip, anomaly_score=score)
        triggered.add(ip)

    print(f"\nTotal devices isolated: {len(triggered)}")

    # --- Print final network state ---
    rollback.status()
    print(f"Rollback log saved to: {rollback.log_path}")


if __name__ == "__main__":
    main()
