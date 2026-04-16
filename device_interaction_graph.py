import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from collections import defaultdict

REQUIRED_COLS = {"src": ["ip.src"], "dst": ["ip.dst"]}

OPTIONAL_COLS = {
    "bytes":      ["frame.len"],
    "proto":      ["ip.proto"],
    "src_port":   ["tcp.srcport"],
    "dst_port":   ["tcp.dstport"],
    "timestamp":  ["frame.time_relative"],
    "mqtt_type":  ["mqtt.msgtype"],
    "label":      ["device_label"],
}

PROTO_MAP = {6: "TCP", 17: "UDP", 1: "ICMP"} 


def _find_col(df, candidates):
    """Return the first candidate column that exists in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _resolve_cols(df):
    """Build a dict of role → actual column name for the given DataFrame."""
    cols = {}
    for role, candidates in {**REQUIRED_COLS, **OPTIONAL_COLS}.items():
        col = _find_col(df, candidates)
        cols[role] = col
    return cols

def build_graph(df, include_timestamps=False):
    """
    Build a directed NetworkX graph from a packet-level DataFrame.

    Parameters
    ----------
    df                : pandas DataFrame with at least ip.src and ip.dst columns
    include_timestamps: whether to store per-packet timestamps on edges (memory-heavy)

    Returns
    -------
    G : nx.DiGraph
        Nodes carry: label (friendly name if available), packet_count, total_bytes
        Edges carry: packet_count, total_bytes, avg_bytes, protocols,
                     src_ports, dst_ports, (timestamps if requested)
    """
    cols = _resolve_cols(df)

    for role in ("src", "dst"):
        if cols[role] is None:
            raise ValueError(
                f"Could not find a '{role}' IP column. "
                f"Expected one of: {REQUIRED_COLS[role]}"
            )

    src_col = cols["src"]
    dst_col = cols["dst"]

    label_lookup = {}
    if cols["label"] is not None:
        tmp = df[[src_col, cols["label"]]].dropna().drop_duplicates()
        label_lookup = dict(zip(tmp[src_col], tmp[cols["label"]]))

    G = nx.DiGraph()

    edge_data = defaultdict(lambda: {
        "packet_count": 0,
        "total_bytes":  0,
        "protocols":    set(),
        "src_ports":    set(),
        "dst_ports":    set(),
        "timestamps":   [] if include_timestamps else None,
    })

    node_packets = defaultdict(int)
    node_bytes   = defaultdict(int)

    col_idx = {name: df.columns.get_loc(name) for name in df.columns}

    src_idx  = col_idx[src_col]
    dst_idx  = col_idx[dst_col]
    byte_idx = col_idx.get(cols["bytes"])
    proto_idx = col_idx.get(cols["proto"])
    sport_idx = col_idx.get(cols["src_port"])
    dport_idx = col_idx.get(cols["dst_port"])
    ts_idx    = col_idx.get(cols["timestamp"]) if include_timestamps else None

    arr = df.values

    for row in arr:
        src = row[src_idx]
        dst = row[dst_idx]

        if pd.isna(src) or pd.isna(dst):
            continue

        src, dst = str(src), str(dst)
        key = (src, dst)
        ed  = edge_data[key]
        ed["packet_count"] += 1
        node_packets[src]  += 1

        if byte_idx is not None:
            b = row[byte_idx]
            try:
                b = float(b)
            except (TypeError, ValueError):
                b = 0
            ed["total_bytes"] += b
            node_bytes[src]   += b

        if proto_idx is not None:
            p = row[proto_idx]
            if p is not None and not (isinstance(p, float) and pd.isna(p)):
                try:
                    ed["protocols"].add(PROTO_MAP.get(int(p), str(p)))
                except (TypeError, ValueError):
                    ed["protocols"].add(str(p))

        if sport_idx is not None:
            sp = row[sport_idx]
            if sp is not None and not (isinstance(sp, float) and pd.isna(sp)):
                try:
                    ed["src_ports"].add(int(sp))
                except (TypeError, ValueError):
                    pass

        if dport_idx is not None:
            dp = row[dport_idx]
            if dp is not None and not (isinstance(dp, float) and pd.isna(dp)):
                try:
                    ed["dst_ports"].add(int(dp))
                except (TypeError, ValueError):
                    pass

        if ts_idx is not None:
            ts = row[ts_idx]
            if ts is not None:
                ed["timestamps"].append(ts)

    all_ips = set(node_packets.keys()) | {k[1] for k in edge_data}
    for ip in all_ips:
        G.add_node(ip,
                   label=label_lookup.get(ip, ip),
                   packet_count=node_packets.get(ip, 0),
                   total_bytes=node_bytes.get(ip, 0))

    for (src, dst), ed in edge_data.items():
        pc = ed["packet_count"]
        tb = ed["total_bytes"]
        attrs = {
            "packet_count": pc,
            "total_bytes":  tb,
            "avg_bytes":    round(tb / pc, 2) if pc else 0,
            "protocols":    list(ed["protocols"]),
            "src_ports":    sorted(ed["src_ports"]),
            "dst_ports":    sorted(ed["dst_ports"]),
        }
        if include_timestamps:
            attrs["timestamps"] = ed["timestamps"]
        G.add_edge(src, dst, **attrs)

    return G

def load_and_build(source, dataset_name="dataset", include_timestamps=False):
    """
    Load a CSV (or accept a DataFrame) and build the interaction graph.

    Parameters
    ----------
    source           : str (file path) or pandas DataFrame
    dataset_name     : label used in print output
    include_timestamps: passed through to build_graph

    Returns
    -------
    G   : nx.DiGraph
    df  : the loaded DataFrame
    """
    if isinstance(source, str):
        print(f"\n[{dataset_name}] Loading: {source}")
        df = pd.read_csv(source, low_memory=False)
        print(f"[{dataset_name}] Rows: {len(df):,}  Columns: {len(df.columns)}")
    else:
        df = source

    df = df.dropna(subset=["ip.src", "ip.dst"])

    print(f"[{dataset_name}] Building graph...")
    G = build_graph(df, include_timestamps=include_timestamps)
    print(f"[{dataset_name}] Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")
    return G, df

def print_summary(G, dataset_name=""):
    label = f" [{dataset_name}]" if dataset_name else ""
    print(f"\n{'='*60}")
    print(f"Graph Summary{label}")
    print(f"{'='*60}")
    print(f"  Nodes (devices) : {G.number_of_nodes()}")
    print(f"  Edges (flows)   : {G.number_of_edges()}")

    top = sorted(G.nodes(data=True), key=lambda x: x[1].get("packet_count", 0), reverse=True)[:5]
    print(f"\n  Top 5 devices by packets sent:")
    for ip, data in top:
        friendly = data.get("label", ip)
        print(f"    {friendly:35s} packets={data.get('packet_count',0):6d}  "
              f"bytes={data.get('total_bytes',0):10,.0f}")

    top_edges = sorted(G.edges(data=True), key=lambda x: x[2].get("packet_count", 0), reverse=True)[:5]
    print(f"\n  Top 5 edges by packet count:")
    for src, dst, data in top_edges:
        src_label = G.nodes[src].get("label", src)
        dst_label = G.nodes[dst].get("label", dst)
        print(f"    {src_label} → {dst_label}")
        print(f"      packets={data['packet_count']}  "
              f"avg_bytes={data['avg_bytes']}  "
              f"protocols={data['protocols']}")
    print(f"{'='*60}\n")

def visualize_graph(G, title="Device Interaction Graph", output_path=None,
                    highlight_nodes=None, highlight_color="red"):
    """
    Draw the interaction graph.

    Parameters
    ----------
    G               : nx.DiGraph from build_graph()
    title           : plot title
    output_path     : if given, save figure to this path (PNG)
    highlight_nodes : set/list of IP addresses to highlight (e.g. anomalous ones)
    highlight_color : color used for highlighted nodes
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    pos = nx.spring_layout(G, seed=42, k=2.5)

    import math
    sizes = []
    for n in G.nodes():
        pc = G.nodes[n].get("packet_count", 1)
        sizes.append(300 + math.log1p(pc) * 120)

    highlight_nodes = set(highlight_nodes or [])
    colors = [highlight_color if n in highlight_nodes else "#4C9BE8" for n in G.nodes()]

    max_packets = max((d.get("packet_count", 1) for _, _, d in G.edges(data=True)), default=1)
    edge_widths = [
        1 + 4 * (d.get("packet_count", 1) / max_packets)
        for _, _, d in G.edges(data=True)
    ]

    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5,
                           edge_color="#888888", arrows=True,
                           arrowsize=15, connectionstyle="arc3,rad=0.1", ax=ax)

    labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
    labels = {n: (v if len(v) <= 20 else v[:17] + "…") for n, v in labels.items()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    patches = [mpatches.Patch(color="#4C9BE8", label="Normal device")]
    if highlight_nodes:
        patches.append(mpatches.Patch(color=highlight_color, label="Highlighted device"))
    ax.legend(handles=patches, loc="upper left")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved graph image: {output_path}")
    plt.show()
    plt.close()

def compare_graphs(G_clean, G_anomaly, label_clean="Clean", label_anomaly="Anomaly"):
    """
    Print a side-by-side comparison of two graphs.
    Highlights nodes/edges that are new or have significantly changed traffic.
    """
    print(f"\n{'='*60}")
    print(f"Graph Comparison: {label_clean} vs {label_anomaly}")
    print(f"{'='*60}")

    clean_nodes  = set(G_clean.nodes())
    anomaly_nodes = set(G_anomaly.nodes())

    new_nodes     = anomaly_nodes - clean_nodes
    missing_nodes = clean_nodes  - anomaly_nodes

    print(f"  Nodes in {label_clean}  : {len(clean_nodes)}")
    print(f"  Nodes in {label_anomaly}: {len(anomaly_nodes)}")
    if new_nodes:
        print(f"  NEW nodes (in anomaly only)    : {new_nodes}")
    if missing_nodes:
        print(f"  MISSING nodes (dropped/silent) : {missing_nodes}")

    clean_edges   = set(G_clean.edges())
    anomaly_edges = set(G_anomaly.edges())

    new_edges     = anomaly_edges - clean_edges
    missing_edges = clean_edges   - anomaly_edges

    print(f"\n  Edges in {label_clean}  : {len(clean_edges)}")
    print(f"  Edges in {label_anomaly}: {len(anomaly_edges)}")
    if new_edges:
        print(f"  NEW edges (anomaly only)      : {new_edges}")
    if missing_edges:
        print(f"  MISSING edges (gone in anomaly): {missing_edges}")

    changed = []
    for src, dst in clean_edges & anomaly_edges:
        c_count = G_clean[src][dst].get("packet_count", 0)
        a_count = G_anomaly[src][dst].get("packet_count", 0)
        if c_count > 0:
            ratio = abs(a_count - c_count) / c_count
            if ratio > 0.5:
                changed.append((src, dst, c_count, a_count, round(ratio * 100)))

    if changed:
        print(f"\n  Edges with >50% change in packet count:")
        for src, dst, cc, ac, pct in sorted(changed, key=lambda x: -x[4]):
            print(f"    {src} → {dst}  clean={cc}  anomaly={ac}  change={pct}%")

    print(f"{'='*60}\n")
    return new_nodes, missing_nodes, new_edges, missing_edges

if __name__ == "__main__":
    import sys

    BASE = "Dataset"

    datasets = {
        "Clean":        os.path.join(BASE, "environmentMonitoring.csv"),
        "Integrity":    os.path.join(BASE, "integrity_dataset3.csv"),
        "Availability": os.path.join(BASE, "availability_dataset3.csv"),
    }

    graphs = {}
    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"[SKIP] {name}: file not found at {path}")
            continue
        G, _ = load_and_build(path, dataset_name=name)
        print_summary(G, dataset_name=name)
        graphs[name] = G

    os.makedirs("graph_output", exist_ok=True)
    for name, G in graphs.items():
        visualize_graph(
            G,
            title=f"Device Interaction Graph – {name}",
            output_path=f"graph_output/{name.lower()}_graph.png"
        )

    if "Clean" in graphs and "Integrity" in graphs:
        compare_graphs(graphs["Clean"], graphs["Integrity"],
                       label_clean="Clean", label_anomaly="Integrity")

    if "Clean" in graphs and "Availability" in graphs:
        compare_graphs(graphs["Clean"], graphs["Availability"],
                       label_clean="Clean", label_anomaly="Availability")