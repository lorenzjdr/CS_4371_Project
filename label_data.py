"""
Identifies and labels devices in the IoT ICU network by analyzing network traffic patterns.
Devices are identified through unique IP addresses and MQTT client IDs.
"""

import pandas as pd

def main():
    print("\nLoading environment (normal) dataset...")
    env_df = pd.read_csv('Dataset/environmentMonitoring.csv') # hardcoded for now
    print(f"Loaded {len(env_df)} records from environment dataset")
    
    try:
        print("\nLoading attack dataset...")
        attack_df = pd.read_csv('Dataset/Attack.csv') # harcoded for now
        print(f"Loaded {len(attack_df)} records from attack dataset")
    except Exception as e:
        print(f"Warning: Could not load attack dataset: {e}")
        attack_df = None


if __name__ == '__main__':
    main()