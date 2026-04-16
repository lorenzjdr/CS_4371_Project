"""
Identifies and labels devices in the IoT ICU network by analyzing network traffic patterns.
Devices are identified through unique IP addresses and MQTT client IDs.
"""

import pandas as pd
from collections import defaultdict
import pickle

def identify_devices(env_df, gateway_ip=None):
    """
    Identify unique devices from network traffic data.
    
    Args:
        env_df: Environment (normal) dataset DataFrame
        gateway_ip: The gateway IP address
        
    Returns:
        device_mapping: Dictionary mapping IP addresses to device labels
        device_stats: Statistics about each device
    """
    
    # Auto-detect gateway if not provided
    if gateway_ip is None:
        gateway_ip = detect_gateway(env_df)
    
    # Get unique source IPs (excluding the gateway)
    unique_ips = env_df['ip.src'].unique()
    unique_ips = sorted([ip for ip in unique_ips if ip != gateway_ip])
    
    print(f"\n{'='*80}")
    print(f"DEVICE IDENTIFICATION & LABELING")
    print(f"{'='*80}")
    print(f"\nFound {len(unique_ips)} unique devices (excluding gateway {gateway_ip})") 
    print(f"Gateway IP: {gateway_ip}\n")
    
    # Assign labels to devices
    # With 2 beds and 10 devices per bed (9 sensors + 1 control unit)
    # We expect 20 devices total 
    
    # NOTE: Bed assignment is based on sequential sorting of IPs
    # This assumes IoT-Flock assigned sequential IPs within each bed
    # No bed info is available in the network traffic data itself
    # (MQTT topics and client IDs are generic/random)
    
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

def detect_gateway(df):
    """
    Auto-detect the gateway by finding the IP that communicates with the most other devices.
    The gateway is the central hub that all IoT devices connect to.
    
    Args:
        df: DataFrame with 'ip.src' and 'ip.dst' columns
        
    Returns:
        gateway_ip: The detected gateway IP address
    """
    # Count how many unique IPs connect TO each IP (as destination)
    ip_connectivity = df['ip.dst'].value_counts()
    
    # Count unique source IPs for each destination IP
    unique_sources_per_dest = df.groupby('ip.dst')['ip.src'].nunique().sort_values(ascending=False)
    
    # The gateway should have connections from many different source IPs
    gateway_ip = unique_sources_per_dest.idxmax()
    num_connected_devices = unique_sources_per_dest.max()
    
    print(f"Detected Gateway: {gateway_ip} (connected to {num_connected_devices} unique devices)")
    return gateway_ip

def main():
    print("\nLoading environment (normal) dataset...")
    env_df = pd.read_csv('Dataset/environmentMonitoring.csv')
    print(f"Loaded {len(env_df)} records from environment dataset")
    
    # Auto-detect gateway
    gateway_ip = detect_gateway(env_df)

    # Identify devices
    device_mapping, device_stats = identify_devices(env_df, gateway_ip)

    # Label the environment dataset
    env_df_labeled = label_dataset(env_df, device_mapping)
    # Save labeled environment dataset
    env_df_labeled.to_csv('Dataset/environmentMonitoring_labeled.csv', index=False)
    print("Saved labeled environment dataset: Dataset/environmentMonitoring_labeled.csv\n")
    
    # serialize data and saved it 
    with open('device_mapping.pkl', 'wb') as f:
        pickle.dump({'device_mapping': device_mapping, 'device_stats': dict(device_stats)}, f)
    print("Saved device mapping (pickle): device_mapping.pkl\n")

if __name__ == '__main__':
    main()