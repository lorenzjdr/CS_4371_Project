# label_data.py

Identifies and labels devices in IoT networks by analyzing network traffic patterns.

## Key Functions

- **`identify_devices()`** - Main function that identifies unique devices from network traffic, classifies them by type, and generates "intelligent" labels
- **`detect_gateway()`** - Detects the network gateway using multi-signal analysis (inbound/outbound connectivity, IoT port usage, bidirectional communication)
- **`analyze_traffic_patterns()`** - Analyzes packet frequency, size distribution, and behavior to classify device types (sensors, control units, monitors, etc.)
- **`cluster_similar_devices()`** - Groups devices with similar traffic characteristics
- **`generate_smart_labels()`** - Creates meaningful device labels combining MQTT IDs and traffic patterns
  - Generate labels:
    - Uses MQTT client ID hash if available: Sensor-743
    - Falls back to type + number if no MQTT ID: HighFreq-Sensor-1
- **`label_dataset()`** - Adds device labels to a dataset based on IP addresses

## Usage

```python
# Load network traffic data
env_df = pd.read_csv('Dataset/environmentMonitoring.csv')

# Auto-detect gateway and identify devices
device_mapping, device_stats, device_profiles = identify_devices(env_df, gateway_ip=None)

# Label the dataset
env_df_labeled = label_dataset(env_df, device_mapping)
```

## Output

- Labeled CSV dataset with device labels
- Pickled device mapping dictionary for later use if needed
- Console output showing device identification, clustering, and labeling process

## Dependencies

- pandas
- numpy
- pickle
