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

#1 Label the data correctly so we can train it

#2 train the model to detect anomalies like the research paper

#3 validate correctness against the attack data
