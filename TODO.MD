## Overview

This project implements an anomaly detection system for an IoT-based ICU with 2 beds, each equipped with:
- **9 patient monitoring sensors/devices** (e.g., heart rate monitor, oxygen sensor, etc.)
- **1 control unit** (Bedx-Control-Unit)

**Total: 20 devices per ICU configuration**

The system uses network traffic analysis (MQTT protocol over TCP) to detect anomalies in device behavior.

# Data Structures

Two Datasets 
- One normal enviroment.csv
- One attack 

The datasets contain network packet captures with features including:
- **IP Layer**: source/destination IPs, protocol, TTL
- **TCP Layer**: source/destination ports, flags, sequence numbers
- **MQTT Layer**: client IDs, topics, message types
- **Timing**: frame deltas, frame timing information

# 1 Label the data correctly (hopefully) so we can train it  
```label_data.py```  
This file labels IoT devices in network traffic data by their IP addresses. It:


Identifies devices — Maps unique source IPs to device names (e.g., "Bed1-Device1", "Bed2-Control-Unit") (might need to modify it by mqtt.topic column later but this is ok for now)
Collects statistics — Packets, bytes, MQTT IDs, and ports for each device
Labels datasets — Adds device_label column to environment and attack traffic CSVs
Compares patterns — Shows normal vs. attack traffic differences per device

# 2 train the model to detect anomalies like the research paper
The goal being generalize the way they did it.

Auto-detects non-numerical features instead of hardcoded hex mappings
Saves model metadata for later validation
Works with any labeled CSV that has a class column

We can use [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)   

- [x] Load the labeled data as a path
- [x] Clean the data
  - [x] Drop rows
  - [x] remove rows with Na
- [ ] Encode non numerical features like they did
- [ ] split the data 
- [ ] Train models use IsolationForest for anomaly detection
- [ ] evaluate
- [ ] save model as .pkl to be used for testing

# 3 validate correctness against the attack data
We need to test the model and it's accuracies against the attack data.
Since we have 2 data sets we can use either one or both 
Also need help on this

# 4 find a way to roll back
Need help on this 

Train model on normal device behavior, then identify when devices act abnormally