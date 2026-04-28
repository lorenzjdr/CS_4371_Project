# Anomaly Detection — Graph Explanations

## Overview
These graphs visualize the performance of three anomaly detection models tested on network traffic data from IoT devices. All models were trained and tested on the same datasets (`integrity_dataset3`, `integrity_dataset5`, `integrity_dataset50`) so their results can be directly compared.

The three models are:
- **OG Random Forest** — the original Random Forest model from the reference code
- **Our Random Forest** — our custom implementation of Random Forest
- **Isolation Forest** — an unsupervised model we built to compare against the Random Forest models

---

## Graph 1 — Model Performance Comparison
**File:** `chart_output/1_model_comparison.png`

This grouped bar chart compares all three models across four standard classification metrics measured on the same test data.

### Metrics explained
| Metric | What it measures |
|---|---|
| **Accuracy** | Out of all predictions, how many were correct |
| **Precision** | Out of all anomalies flagged, how many were actually anomalies |
| **Recall** | Out of all real anomalies, how many did the model catch |
| **F1 Score** | A combined score balancing Precision and Recall |

### How to read it
- Each group of bars represents one metric
- The three colored bars within each group represent the three models
- Higher is better for all metrics (max score = 1.0)
- Comparing bar heights within a group shows which model performed best on that metric

### Key takeaway
Both Random Forest models score perfectly (1.0) across all metrics on this dataset, meaning they correctly identified every anomaly with no false positives. The Isolation Forest scores lower because it is unsupervised — it was never shown labeled examples during training and had to learn what "normal" looks like on its own.

---

## Graph 2 — Our Random Forest: Top 10 Feature Importances
**File:** `chart_output/2_rf_feature_importances.png`

This horizontal bar chart shows which data columns (features) our Random Forest model relied on most when deciding whether a sample was an anomaly.

### How to read it
- Each bar represents one feature (column) from the dataset
- The longer the bar, the more that feature influenced the model's decisions
- The scores shown are importance values — they represent each feature's share of contribution across all decision trees in the model
- Features are sorted from least to most important (top = most important)

### Key features and what they mean
| Feature | Why it matters |
|---|---|
| **tcp.ack** | TCP acknowledgment numbers track packet sequencing — integrity attacks often disrupt this |
| **tcp.checksum** | Checksums verify data has not been tampered with — corrupted packets show unusual checksum values |
| **frame.len** | Packet size — injected or modified packets often have abnormal lengths |
| **frame.time_relative** | Timing between packets — attacks can disrupt normal traffic timing patterns |
| **tcp.pdu.size** | Size of the protocol data unit — another indicator of packet manipulation |
| **mqtt.hdrflags / mqtt.msgtype / mqtt.len** | MQTT application-layer fields — less important than TCP fields but still contribute to detection |

### Key takeaway
The model primarily detects anomalies using low-level TCP and packet size characteristics rather than application-layer MQTT data. This makes sense for an integrity dataset where attacks corrupt packet contents at the transport layer.