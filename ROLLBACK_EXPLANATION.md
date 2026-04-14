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

---

## Stage 3 — `randomforest.py` (Random Forest)

This is a separate **supervised** approach — it uses the `label` column (normal/attack) to train a classifier.

Unlike Isolation Forest, it needs labeled data to learn from. It is good at classifying known attack patterns but will not catch novel ones. The Isolation Forest approach is more generalizable.

---

## Stage 4 — `detect_and_rollback.py` + `rollback.py`

This is where detection meets response.

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

### `detect_and_rollback.py` — The Runner

1. Loads the trained Isolation Forest model
2. Loads `device_mapping.pkl` to know which IPs are which devices
3. Detects the gateway from the environment CSV
4. Builds the `NetworkGraph` from the device mapping
5. Loads `Attack_labeled.csv` and prepares its features the same way as training
6. Runs `model.predict()` — returns `-1` (anomaly) or `1` (normal) per packet
7. For each anomalous packet's `ip.src`:
   - If the IP is a known device → isolate it
   - If the IP is unknown (external attacker) → register it as an `Attacker` node first, then isolate it
   - Only triggers once per unique IP

---

## Why 2 Attacker IPs, Not 20 Device IPs?

The attack dataset simulates **external attackers** (`192.168.1.90`, `192.168.1.91`) sending malicious packets *into* the ICU network. The legitimate ICU devices (`10.5.126.x`) are not the ones misbehaving — the threat is coming from outside.

If the attack were an **inside threat** (a compromised ICU device acting abnormally), you would see one of the `Bed` devices flagged and isolated instead.

---

## Data Flow Summary

```
environmentMonitoring.csv  ──→  train_model.py  ──→  isolation_forest_model.pkl
                                                              │
device_mapping.pkl  ─────────────────────────────────────────┤
                                                              ▼
Attack_labeled.csv  ──────────────────────────────→  detect_and_rollback.py
                                                              │
                                                    model.predict() → anomalous IPs
                                                              │
                                                    rollback.trigger(ip)
                                                              │
                                                    edges removed from NetworkGraph
                                                              │
                                                    rollback_log.json
```
