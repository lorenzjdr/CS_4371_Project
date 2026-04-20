# IoT-AD: System Explanation

## Overview

The project is an **IoT Anomaly Detection + Response** system for a simulated ICU network. It has 4 stages:

```
1. label_data.py          → identify devices from network traffic
2. train_model.py         → train a model on normal behavior
3. randomforest.py        → (separate) train a classifier on labeled data
4. detect_and_rollback.py → detect anomalies + isolate bad devices
```

---

## Stage 1 — `label_data.py`

The raw CSV files are packet captures (like Wireshark dumps) with columns like `ip.src`, `ip.dst`, `frame.len`, `mqtt.clientid`, etc.

There are no device names in the data — just IP addresses. So this script:

1. **Finds the gateway** — the IP that the most other IPs talk to (the MQTT broker)
2. **Sorts remaining IPs** — assigns labels like `Bed1-Device1`, `Bed2-Control-Unit` based on sequential order
3. **Saves `device_mapping.pkl`** — a dict of `{ip: label}` for use by later scripts
4. **Saves labeled CSVs** — adds a `device_label` column to the environment and attack datasets

---

## Stage 2 — `train_model.py` (Isolation Forest)

The **Isolation Forest** is an unsupervised anomaly detection algorithm. You train it only on *normal* data — it learns what normal looks like.

How it works internally: it builds random decision trees that try to isolate individual data points. Anomalies are easier to isolate (fewer splits needed), so they get lower scores.

This script:
1. Loads the normal environment CSV
2. Encodes categorical columns (like IP strings) into numbers
3. Trains `IsolationForest(contamination=0.001)` — assumes 0.1% of training data might be noise
4. Saves the model to a `.pkl` file

Three model variants are saved depending on `DATASET_CHOICE`:
- `isolation_forest_model_environment.pkl` — trained on environment monitoring only
- `isolation_forest_model_patient.pkl` — trained on patient monitoring only
- `isolation_forest_model_both.pkl` — trained on both datasets combined

---

## Stage 3 — `randomforest.py` (Random Forest)

This is a separate **supervised** approach — it uses the `label` column (normal/attack) to train a classifier.

Unlike Isolation Forest, it needs labeled data to learn from. It is good at classifying known attack patterns but will not catch novel ones. The Isolation Forest approach is more generalizable.

---

## Stage 4 — `detect_and_rollback.py` + `rollback.py`

This is where detection meets response.

### `anomaly_data/anomaly.py` — Synthetic Anomaly Generation

Before detection runs, `anomaly.py` generates two synthetic anomaly datasets from the normal environment data to simulate insider/device-level threats:

- **Integrity anomalies** (`integrity_dataset5.csv`) — a small number of rows have numeric columns (e.g. `frame.len`, `tcp.checksum`) inflated by a random value, simulating a device sending corrupted sensor readings
- **Availability anomalies** (`availability_dataset5.csv`) — a small number of rows are deleted, simulating packet loss or a device going silent

These datasets contain the real ICU device IPs (`10.5.126.x`) so when anomalies are detected, named bed devices are isolated rather than unknown external nodes.

---

### `rollback.py` — Two Classes

**`NetworkGraph`**
- Wraps a `networkx.DiGraph` (directed graph)
- Each device = a node, each communication path = an edge
- Built as a **star topology**: every device has 2 edges (device→gateway, gateway→device) because all IoT devices talk through the MQTT broker
- Example after build:

```
Bed1-Device1 ──→ Gateway
Gateway      ──→ Bed1-Device1
Bed1-Device2 ──→ Gateway
...
```

**`RollbackManager`**
- Holds a reference to the graph
- `trigger(ip)` — called when an anomaly is detected:
  1. Logs `ANOMALY_DETECTED` event
  2. Calls `_isolate(ip)`:
     - Saves the device's edges to the `isolated_devices` dict
     - Removes all edges from the graph
     - Sets node status to `"isolated"`
     - Logs `DEVICE_ISOLATED` event
- `restore(ip)` — admin manually calls this after fixing the device:
  - Re-adds the saved edges back to the graph
  - Sets status back to `"active"`
- All events are written to `rollback_log.json` with timestamps

---

### `detect_and_rollback.py` — The Runner

Runs **three sequential scans** against the same `NetworkGraph` and `RollbackManager`, so isolation state accumulates across all scans.

The core detection logic is in `detect_and_isolate()`, which is reused for each scan:
1. Drops rows missing `ip.src`
2. Encodes and aligns features to match the model's expected input
3. Runs `model.predict()` — returns `-1` (anomaly) or `1` (normal) per packet
4. For each anomalous packet's `ip.src`:
   - If the IP is a known device → isolate it directly
   - If the IP is unknown (external attacker) → register it as an `Attacker` node connected to the gateway, then isolate it
   - Only triggers once per unique IP per scan; already-isolated devices are skipped

**Scan 1 — External Attack Data (`Attack_labeled.csv`)**
- 80,126 packets, all labeled attack traffic
- Source IPs are external (`192.168.1.90`, `192.168.1.91`) — not in the device mapping
- Both are registered as `Attacker` nodes and isolated

**Scan 2 — Integrity Anomalies (`integrity_dataset5.csv`)**
- ~31,758 packets, mostly normal with 5 corrupted rows injected by `anomaly.py`
- Source IPs are real ICU devices (`10.5.126.x`) already in the network graph
- **Limitation:** The Isolation Forest does not reliably detect the injected rows. The corruptions are too subtle (e.g. small numeric offsets, a string suffix) to stand out against the natural variation in 31,758 packets. The model flags 0 of the 5 injected rows and instead flags ~454 natural statistical outliers that already exist in the normal data.
- The named bed devices isolated in this scan are **false positives**, not the actual corrupted-data sources

**Scan 3 — Availability Anomalies (`availability_dataset5.csv`)**
- ~31,753 packets (5 rows deleted from the normal dataset)
- Row deletion is **not directly detectable** by Isolation Forest — the model scores individual rows and cannot observe that rows are missing
- Any devices flagged here are the same natural outliers as Scan 2; they are skipped as already isolated

---

## Gateway Protection

The gateway is explicitly excluded from isolation. Because all devices route traffic through the gateway (the MQTT broker), isolating it would sever every device from the network at once — making the rollback response too broad to be useful.

Instead, only individual end devices are isolated when flagged. The gateway stays online throughout all scans so that non-flagged devices can continue communicating normally.

---

## Data Flow Summary

```
environmentMonitoring.csv  ──→  train_model.py  ──→  isolation_forest_model_environment.pkl
                                                                  │
anomaly.py  ──→  integrity_dataset5.csv  ───────────────────────┐ │
            └──→  availability_dataset5.csv  ───────────────────┤ │
                                                                 │ │
device_mapping.pkl  ─────────────────────────────────────────┐  │ │
                                                             ▼  ▼ ▼
Attack_labeled.csv  ──────────────────────────────→  detect_and_rollback.py
                                                              │
                                               detect_and_isolate() × 3 scans
                                                              │
                                                    rollback.trigger(ip)
                                                              │
                                                    edges removed from NetworkGraph
                                                              │
                                                    rollback_log.json
```
