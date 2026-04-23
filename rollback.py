"""
Rollback Mechanism for IoT-AD
Graph-based device isolation triggered on anomaly detection.

Devices are represented as nodes, communication paths as edges.
When an anomaly is detected, the affected device's edges are removed
to prevent anomaly propagation across the network.
"""

import json
import datetime
import networkx as nx


class NetworkGraph:
    """Represents IoT devices as nodes and communication paths as directed edges."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_device(self, device_id, **attrs):
        self.graph.add_node(device_id, status="active", **attrs)

    def add_comm_path(self, src, dst):
        self.graph.add_edge(src, dst)

    def get_neighbors(self, device_id):
        """Return all devices directly connected to this device (in or out)."""
        predecessors = list(self.graph.predecessors(device_id))
        successors = list(self.graph.successors(device_id))
        return list(set(predecessors + successors))

    def build_from_device_mapping(self, device_mapping, gateway_ip):
        """
        Populate the graph from label_data.py's device_mapping dict.
        Uses a star topology: every device connects to/from the gateway (MQTT broker).

        Args:
            device_mapping: dict of {ip: label} from device_mapping.pkl
            gateway_ip: the detected gateway/broker IP
        """
        self.add_device(gateway_ip, label="Gateway")
        for ip, label in device_mapping.items():
            self.add_device(ip, label=label)
            self.add_comm_path(ip, gateway_ip)
            self.add_comm_path(gateway_ip, ip)


class RollbackManager:
    """
    Manages isolation and restoration of compromised IoT devices.

    On anomaly detection:
      1. Marks the device as compromised
      2. Removes all its communication edges (isolates it)
      3. Logs the event to a JSON audit log

    Restoration is a manual admin action after remediation.
    """

    def __init__(self, network_graph: NetworkGraph, log_path="rollback_log.json"):
        self.graph = network_graph
        self.log_path = log_path
        self.isolated_devices = {}  # ip -> list of removed edges
        self._load_log()

    def _load_log(self):
        try:
            with open(self.log_path, "r") as f:
                self.log = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.log = []

    def _save_log(self):
        with open(self.log_path, "w") as f:
            json.dump(self.log, f, indent=2)

    def _log_event(self, event_type, device_id, details=""):
        label = self.graph.graph.nodes[device_id].get("label", "unknown")
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": event_type,
            "device": device_id,
            "label": label,
            "details": details,
        }
        self.log.append(entry)
        self._save_log()
        print(f"[ROLLBACK] {entry['timestamp']} | {event_type:20} | {label} ({device_id})")

    def trigger(self, device_id, anomaly_score=None):
        """
        Called when the anomaly detector flags a device.
        Marks the device as compromised and isolates it from the network.

        Args:
            device_id: IP address of the anomalous device
            anomaly_score: optional score from the model for logging

        Returns:
            True if isolation was performed, False otherwise
        """
        if device_id not in self.graph.graph.nodes:
            print(f"[ROLLBACK] Unknown device: {device_id} — skipping")
            return False

        if device_id in self.isolated_devices:
            print(f"[ROLLBACK] Device already isolated: {device_id} — skipping")
            return False

        details = f"anomaly_score={anomaly_score:.4f}" if anomaly_score is not None else ""
        self._log_event("ANOMALY_DETECTED", device_id, details)
        self._isolate(device_id)
        return True

    def _isolate(self, device_id):
        """Remove all communication edges for this device."""
        outgoing = list(self.graph.graph.out_edges(device_id))
        incoming = list(self.graph.graph.in_edges(device_id))
        removed_edges = outgoing + incoming

        self.graph.graph.remove_edges_from(removed_edges)
        self.graph.graph.nodes[device_id]["status"] = "isolated"
        self.isolated_devices[device_id] = removed_edges

        affected = set([e[0] for e in incoming] + [e[1] for e in outgoing])
        affected.discard(device_id)
        self._log_event(
            "DEVICE_ISOLATED",
            device_id,
            f"removed {len(removed_edges)} edges | affected neighbors: {sorted(affected)}"
        )

    def restore(self, device_id):
        """
        Restore a previously isolated device after admin remediation.
        Re-adds all communication edges that were removed during isolation.

        Args:
            device_id: IP address of the device to restore

        Returns:
            True if restored, False if device was not isolated
        """
        if device_id not in self.isolated_devices:
            print(f"[ROLLBACK] Device not currently isolated: {device_id}")
            return False

        edges = self.isolated_devices.pop(device_id)
        self.graph.graph.add_edges_from(edges)
        self.graph.graph.nodes[device_id]["status"] = "active"
        self._log_event("DEVICE_RESTORED", device_id, f"restored {len(edges)} edges")
        return True

    def status(self):
        """Print a summary of the current network state."""
        print(f"\n{'='*65}")
        print("NETWORK STATUS")
        print(f"{'='*65}")
        for node, data in sorted(self.graph.graph.nodes(data=True), key=lambda x: x[0]):
            label = data.get("label", node)
            node_status = data.get("status", "active")
            edge_count = self.graph.graph.degree(node)
            flag = "  [ISOLATED]" if node_status == "isolated" else ""
            print(f"  {label:35} | edges: {edge_count:2}{flag}")
        print(f"{'='*65}")
        print(f"  Isolated devices: {len(self.isolated_devices)}")
        print(f"{'='*65}\n")
