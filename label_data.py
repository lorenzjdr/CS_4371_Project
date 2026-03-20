"""
Identifies and labels devices in the IoT ICU network by analyzing network traffic patterns.
Devices are identified through unique IP addresses and MQTT client IDs.
"""

import pandas as pd
from collections import defaultdict

def identify_devices(env_df, attack_df=None):
    """
    Identify unique devices from network traffic data.
    
    Args:
        env_df: Environment (normal) dataset DataFrame
        attack_df: Optional attack dataset DataFrame for verification
        
    Returns:
        device_mapping: Dictionary mapping IP addresses to device labels
        device_stats: Statistics about each device
    """
    
    # Get unique source IPs (excluding the gateway 10.5.126.84)
    unique_ips = env_df['ip.src'].unique()
    unique_ips = sorted([ip for ip in unique_ips if ip != '10.5.126.84'])
    
    print(f"\n{'='*80}")
    print(f"DEVICE IDENTIFICATION & LABELING")
    print(f"{'='*80}")
    print(f"\nFound {len(unique_ips)} unique devices (excluding gateway 10.5.126.84)") 
    print(f"Gateway IP: 10.5.126.84\n")
    
    # Assign labels to devices
    # With 2 beds and 10 devices per bed (9 sensors + 1 control unit)
    # We expect 20 devices total 
    # it should be at least 20
    device_mapping = {}
    device_stats = defaultdict(lambda: {
        'mqtt_clientids': set(),
        'packet_count': 0,
        'source_ports': set(),
        'avg_packet_size': 0,
        'total_bytes': 0
    })
    
    # go through all the unique ip's
    for idx, ip in enumerate(unique_ips):
        bed_num = (idx // 10) + 1
        device_num = (idx % 10) + 1
        
        # Label control units differently
        # since they don't provide the device names
        # we assgin a generic id for them
        if device_num == 10:
            device_label = f"Bed{bed_num}-Control-Unit"
        else:
            device_label = f"Bed{bed_num}-Device{device_num}"
        
        device_mapping[ip] = device_label
        
        # Collect stats for this device
        device_data = env_df[env_df['ip.src'] == ip]
        device_stats[ip]['packet_count'] = len(device_data)
        device_stats[ip]['total_bytes'] = device_data['frame.len'].sum()
        device_stats[ip]['avg_packet_size'] = device_data['frame.len'].mean()
        
        # Get MQTT client IDs if available
        mqtt_ids = device_data['mqtt.clientid'].unique()
        mqtt_ids = [m for m in mqtt_ids if pd.notna(m) and m != 0]
        device_stats[ip]['mqtt_clientids'] = mqtt_ids
        
        # Get source ports used
        source_ports = device_data['tcp.srcport'].unique()
        device_stats[ip]['source_ports'] = sorted(source_ports.tolist())
    
    print(f"Device Mapping:")
    print(f"{'-'*80}")
    for ip, label in sorted(device_mapping.items(), key=lambda x: (x[1].split('-')[0], x[1])):
        stats = device_stats[ip]
        print(f"  {label:30} | IP: {ip:15} | Packets: {stats['packet_count']:6} | "
              f"Bytes: {stats['total_bytes']:10,.0f}")
        if stats['mqtt_clientids']:
            print(f"    {'':30}   MQTT Client IDs: {stats['mqtt_clientids']}")
    
    print(f"{'-'*80}\n")
    
    return device_mapping, device_stats

def label_dataset(df, device_mapping):
    """
    Add device labels to a dataset based on IP addresses.
    
    Args:
        df: DataFrame with 'ip.src' column
        device_mapping: Dictionary mapping IPs to device labels
        
    Returns:
        DataFrame with new 'device_label' column
    """

    df_copy = df.copy()
    df_copy['device_label'] = df_copy['ip.src'].map(device_mapping)
    return df_copy

def analyze_and_save_attack_data(env_df_labeled, attack_df, device_mapping):
    """
    Label the attack dataset and compare traffic patterns with normal data.
    
    Args:
        env_df_labeled: Labeled environment (normal) dataset DataFrame
        attack_df: Attack dataset DataFrame
        device_mapping: Dictionary mapping IPs to device labels
    """
    attack_df_labeled = label_dataset(attack_df, device_mapping)
    
    # Compare traffic patterns between normal and attack
    print(f"\nTraffic Analysis - Normal vs Attack:")
    print(f"{'-'*80}")
    for device_label in sorted([x for x in env_df_labeled['device_label'].unique() if pd.notna(x)]):
        if pd.isna(device_label):
            continue
        normal_count = len(env_df_labeled[env_df_labeled['device_label'] == device_label])
        attack_count = len(attack_df_labeled[attack_df_labeled['device_label'] == device_label])
        print(f"  {device_label:30} | Normal: {normal_count:6} | Attack: {attack_count:6}")
    print(f"{'-'*80}\n")
    
    # Save labeled attack dataset
    attack_df_labeled.to_csv('Dataset/Attack_labeled.csv', index=False)
    print("Saved labeled attack dataset: Dataset/Attack_labeled.csv\n")
    
    return attack_df_labeled

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

    # Identify devices
    device_mapping, device_stats = identify_devices(env_df, attack_df)

    #print(device_mapping)
    #print(device_stats)

    # Label the environment dataset
    env_df_labeled = label_dataset(env_df, device_mapping)

    # label attack data as well
    if attack_df is not None:
        attack_df_labeled = analyze_and_save_attack_data(env_df_labeled, attack_df, device_mapping)
        print(attack_df_labeled)

if __name__ == '__main__':
    main()