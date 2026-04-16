"""
Identifies and labels devices in the IoT ICU network by analyzing network traffic patterns.
Devices are identified through unique IP addresses and MQTT client IDs.
"""

import pandas as pd
from collections import defaultdict
import pickle
import numpy as np

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
    Intelligently detect the gateway using multiple signals:
    1. Bidirectional communication (both source and destination traffic)
    2. Common IoT ports (1883 for MQTT, 5555 for control, etc.)
    3. Hub-like connectivity pattern (communicates with many devices)
    
    Args:
        df: DataFrame with 'ip.src', 'ip.dst', 'tcp.dstport' columns
        
    Returns:
        gateway_ip: The detected gateway IP address
    """
    MQTT_PORTS = {1883, 8883}  # MQTT standard ports
    CONTROL_PORTS = {5555, 9090, 8080, 8000}  # Common control ports
    IOT_PORTS = MQTT_PORTS | CONTROL_PORTS
    
    print(f"\n{'='*80}")
    print(f"GATEWAY DETECTION (Multi-Signal Analysis)")
    print(f"{'='*80}\n")
    
    # Calculate connectivity metrics for each IP
    inbound_connections = df['ip.dst'].value_counts()  # How many packets TO this IP
    unique_sources = df.groupby('ip.dst')['ip.src'].nunique()  # Unique sources TO this IP
    
    outbound_connections = df['ip.src'].value_counts()  # How many packets FROM this IP
    unique_dests = df.groupby('ip.src')['ip.dst'].nunique()  # Unique destinations FROM this IP
    
    # Analyze port usage for each IP (as destination)
    iot_port_usage = {}
    for ip in df['ip.dst'].unique():
        if pd.isna(ip):
            continue
        port_data = df[df['ip.dst'] == ip]['tcp.dstport']
        iot_ports_used = len([p for p in port_data if pd.notna(p) and int(p) in IOT_PORTS])
        iot_port_usage[ip] = iot_ports_used
    
    # Score each candidate
    candidates = []
    all_ips = set(df['ip.src'].unique()) | set(df['ip.dst'].unique())
    all_ips = [ip for ip in all_ips if pd.notna(ip)]
    
    for ip in all_ips:
        score = 0
        details = []
        
        # Signal 1: High inbound connectivity
        inbound_count = inbound_connections.get(ip, 0)
        unique_source_count = unique_sources.get(ip, 0)
        if unique_source_count > 0:
            score += unique_source_count * 2  # Weight: 2x
            details.append(f"Inbound from {unique_source_count} devices")
        
        # Signal 2: High outbound connectivity
        outbound_count = outbound_connections.get(ip, 0)
        unique_dest_count = unique_dests.get(ip, 0)
        if unique_dest_count > 0:
            score += unique_dest_count
            details.append(f"Outbound to {unique_dest_count} destinations")
        
        # Signal 3: Uses IoT ports
        iot_ports = iot_port_usage.get(ip, 0)
        if iot_ports > 0:
            score += iot_ports * 3  # Weight: 3x (strong signal)
            details.append(f"Uses {iot_ports} IoT ports")
        
        # Signal 4: Bidirectional communication (implies hub)
        if unique_source_count > 0 and unique_dest_count > 0:
            score += 5  # Bonus for bidirectional
            details.append(f"Bidirectional communication")
        
        if score > 0:
            candidates.append({
                'ip': ip,
                'score': score,
                'inbound_devices': unique_source_count,
                'outbound_dests': unique_dest_count,
                'iot_ports_used': iot_ports,
                'total_packets': inbound_count + outbound_count,
                'details': details
            })
    
    # Sort by score
    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    
    # Display top candidates
    print(f"Gateway Candidates (ranked by likelihood):\n")
    for rank, candidate in enumerate(candidates[:5], 1):
        print(f"{rank}. IP: {candidate['ip']}")
        print(f"   Score: {candidate['score']} | Packets: {candidate['total_packets']:,}")
        print(f"   Inbound: {candidate['inbound_devices']} devices | "
              f"Outbound: {candidate['outbound_dests']} destinations | "
              f"IoT Ports: {candidate['iot_ports_used']}")
        for detail in candidate['details']:
            print(f"   * {detail}")
        print()
    
    # Select top candidate
    if not candidates:
        raise ValueError("No gateway candidates found!")
    
    gateway_ip = candidates[0]['ip']
    print(f"{'-'*80}")
    print(f"Selected Gateway: {gateway_ip}\n")
    
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