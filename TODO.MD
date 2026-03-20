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
