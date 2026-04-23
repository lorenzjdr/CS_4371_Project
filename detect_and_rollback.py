"""
Detection + Rollback Runner

Loads the trained Isolation Forest model, runs it against:
  1. External attack data (Attack_labeled.csv)
  2. Integrity anomaly data (synthetic corrupted sensor readings)
  3. Availability anomaly data (note: row-deletion anomalies are not detectable by IF)

Triggers the rollback mechanism for any anomalous devices detected.

Usage:
    python detect_and_rollback.py
"""

import copy
import os
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from rollback import NetworkGraph, RollbackManager
from label_data import detect_gateway


MODEL_PATH = "models/isolation_forest_model_environment.pkl"
DEVICE_MAPPING_PATH = "device_mapping.pkl"
ATTACK_DATA_PATH = "Dataset/Attack_labeled.csv"
ENV_DATA_PATH = "Dataset/environmentMonitoring_labeled.csv"
INTEGRITY_DATA_PATH = "anomaly_data/anomaly_datasets5/integrity_dataset5.csv"


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


def detect_and_isolate(df, scan_label, model, net_graph, rollback, gateway_ip, device_mapping=None):
    """
    Run anomaly detection on a DataFrame and trigger rollback for flagged device IPs.

    Args:
        df: DataFrame with an 'ip.src' column
        scan_label: Human-readable label for this scan (used in print output)
        model: Trained Isolation Forest model
        net_graph: NetworkGraph instance
        rollback: RollbackManager instance
        gateway_ip: Gateway IP for registering unknown attacker nodes
        device_mapping: Optional dict of {ip: label} used when 'device_label' column is absent

    Returns:
        Set of IPs that were isolated during this scan
    """
    print(f"\n--- {scan_label} ---")

    df = df.dropna(subset=["ip.src"]).reset_index(drop=True)
    if df.empty:
        print("  No records to scan.")
        return set()

    print(f"  Records: {len(df)}")
    X = prepare_features(df, model.feature_names_in_)

    scores = model.decision_function(X)   # lower = more anomalous
    preds = model.predict(X)              # -1 = anomaly, 1 = normal

    anomalous_mask = preds == -1
    print(f"  Anomalous packets detected: {anomalous_mask.sum()} / {len(preds)}")

    triggered = set()
    for idx, row in df[anomalous_mask].iterrows():
        ip = row.get("ip.src")
        if pd.isna(ip) or ip in triggered:
            continue

        # Never isolate the gateway — doing so would sever all devices at once
        if ip == gateway_ip:
            continue

        # If the IP is not a known device, register it as an attacker node
        # connected to the gateway so it can still be isolated.
        # Prefer device_mapping lookup over the (potentially absent) device_label column.
        if ip not in net_graph.graph.nodes:
            device_label = (device_mapping or {}).get(ip)
            if device_label is None:
                device_label = row.get("device_label")
            if device_label is None or (isinstance(device_label, float) and pd.isna(device_label)):
                device_label = f"Attacker ({ip})"
            net_graph.add_device(ip, label=device_label)
            net_graph.add_comm_path(ip, gateway_ip)
            net_graph.add_comm_path(gateway_ip, ip)

        rollback.trigger(ip, anomaly_score=float(scores[idx]))
        triggered.add(ip)

    print(f"  Devices isolated this scan: {len(triggered)}")
    return triggered


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

    # Snapshot the graph before any isolation
    graph_before = copy.deepcopy(net_graph.graph)

    all_isolated = set()

    # --- Scan 1: External attack data ---
    attack_df = pd.read_csv(ATTACK_DATA_PATH, low_memory=False)
    isolated = detect_and_isolate(
        attack_df, "Scan 1: External Attack Data", model, net_graph, rollback, gateway_ip, device_mapping
    )
    all_isolated |= isolated

    # --- Scan 2: Integrity anomaly data (corrupted sensor readings) ---
    integrity_df = pd.read_csv(INTEGRITY_DATA_PATH, low_memory=False)
    isolated = detect_and_isolate(
        integrity_df, "Scan 2: Integrity Anomalies (corrupted sensor values)", model, net_graph, rollback, gateway_ip, device_mapping
    )
    all_isolated |= isolated

    # --- Final summary ---
    print(f"\nTotal unique devices isolated across all scans: {len(all_isolated)}")
    rollback.status()
    print(f"Rollback log saved to: {rollback.log_path}")

    # --- Visualize before/after rollback ---
    # Shell layout: gateway in center, all other devices in outer ring
    all_nodes = list(net_graph.graph.nodes())
    gateway_nodes = [n for n in all_nodes if net_graph.graph.nodes[n].get("label") == "Gateway"]
    outer_nodes = [n for n in all_nodes if n not in gateway_nodes]
    pos = nx.shell_layout(net_graph.graph, nlist=[gateway_nodes, outer_nodes])

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor="#1a1a2e")

    for ax, G, title in [
        (axes[0], graph_before, "Before Rollback"),
        (axes[1], net_graph.graph, "After Rollback"),
    ]:
        ax.set_facecolor("#1a1a2e")
        colors = []
        sizes = []
        for node in G.nodes():
            if G.nodes[node].get("label") == "Gateway":
                colors.append("gold")
                sizes.append(1800)
            elif node in all_isolated:
                colors.append("#FF4444")
                sizes.append(900)
            else:
                colors.append("#4C9BE8")
                sizes.append(900)

        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=10,
                               edge_color="#555577", width=1.2, alpha=0.6,
                               connectionstyle="arc3,rad=0.1",
                               min_source_margin=15, min_target_margin=15)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                               node_size=sizes, linewidths=1.2, edgecolors="white")
        labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                                font_size=7, font_color="white", font_weight="bold")
        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=12)
        ax.axis("off")

    # Legend
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color="gold",    label="Gateway"),
        mpatches.Patch(color="#4C9BE8", label="Active device"),
        mpatches.Patch(color="#FF4444", label="Isolated device"),
    ]
    axes[1].legend(handles=legend_patches, loc="lower right",
                   facecolor="#2a2a4a", edgecolor="#555577",
                   labelcolor="white", fontsize=9)

    os.makedirs("graph_output", exist_ok=True)
    output_path = "graph_output/rollback_state.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print(f"Graph saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
