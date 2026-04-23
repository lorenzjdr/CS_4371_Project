import re
from pathlib import Path
from collections import defaultdict

def extract_metrics_from_reports():
    """
    Extract Total rows and Anomalies found from all .txt report files
    in outputs, outputs-ifmb, and outputs-rf directories.
    """
    
    # Base directory
    base_dir = Path(__file__).parent.parent
    output_dirs = [
        base_dir / "outputs",
        base_dir / "outputs-ifmb",
        base_dir / "outputs-rf"
    ]
    
    results = defaultdict(list)
    output_file = Path(__file__).parent / "metrics_report.txt"
    
    def write_output(text):
        """Print to console and write to file"""
        print(text)
        with open(output_file, 'a') as f:
            f.write(text + '\n')
    
    # Clear previous report
    output_file.write_text("")
    
    write_output("="*70)
    write_output("METRICS EXTRACTION REPORT")
    write_output("="*70)
    write_output("")
    
    for output_dir in output_dirs:
        if not output_dir.exists():
            continue
            
        dir_name = output_dir.name
        write_output(f"\n[{dir_name.upper()}]")
        write_output("-" * 70)
        
        # Find all .txt files
        txt_files = sorted(output_dir.glob("*.txt"))
        
        if not txt_files:
            write_output(f"  No .txt files found in {dir_name}")
            continue
        
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                content = f.read()
            
            # Extract Total rows
            total_rows_match = re.search(r'Total rows:\s*(\d+)', content)
            total_rows = int(total_rows_match.group(1)) if total_rows_match else None
            
            # Extract Anomalies found
            anomalies_match = re.search(r'Anomalies found:\s*(\d+)', content)
            anomalies = int(anomalies_match.group(1)) if anomalies_match else None
            
            if total_rows is not None and anomalies is not None:
                anomaly_percentage = (anomalies / total_rows * 100) if total_rows > 0 else 0
                normal_rows = total_rows - anomalies
                
                results[dir_name].append({
                    'file': txt_file.name,
                    'total_rows': total_rows,
                    'anomalies': anomalies,
                    'normal_rows': normal_rows,
                    'anomaly_percentage': anomaly_percentage
                })
                
                write_output(f"  {txt_file.name}")
                write_output(f"    Total rows: {total_rows:,}")
                write_output(f"    Anomalies found: {anomalies:,}")
                write_output(f"    Normal rows: {normal_rows:,}")
                write_output(f"    Anomaly percentage: {anomaly_percentage:.2f}%")
                write_output("")
    
    # Summary Statistics
    write_output("\n" + "="*70)
    write_output("SUMMARY STATISTICS")
    write_output("="*70)
    
    for dir_name in sorted(results.keys()):
        metrics = results[dir_name]
        
        write_output(f"\n{dir_name.upper()}:")
        write_output("-" * 70)
        write_output(f"  Number of reports: {len(metrics)}")
        
        total_rows_all = sum(m['total_rows'] for m in metrics)
        total_anomalies_all = sum(m['anomalies'] for m in metrics)
        total_normal_all = sum(m['normal_rows'] for m in metrics)
        
        avg_anomaly_percent = (total_anomalies_all / total_rows_all * 100) if total_rows_all > 0 else 0
        
        write_output(f"  Total rows (all reports): {total_rows_all:,}")
        write_output(f"  Total anomalies found: {total_anomalies_all:,}")
        write_output(f"  Total normal rows: {total_normal_all:,}")
        write_output(f"  Overall anomaly percentage: {avg_anomaly_percent:.2f}%")
        write_output(f"  Average anomalies per report: {total_anomalies_all / len(metrics):,.0f}")
        write_output(f"  Average anomaly percentage per report: {sum(m['anomaly_percentage'] for m in metrics) / len(metrics):.2f}%")
        write_output(f"  Max anomalies in single report: {max(m['anomalies'] for m in metrics):,}")
        write_output(f"  Min anomalies in single report: {min(m['anomalies'] for m in metrics):,}")
    
    # Comparative Analysis
    if len(results) > 1:
        write_output("\n" + "="*70)
        write_output("COMPARATIVE ANALYSIS")
        write_output("="*70)
        
        comparison_data = {}
        for dir_name in sorted(results.keys()):
            metrics = results[dir_name]
            total_anomalies = sum(m['anomalies'] for m in metrics)
            total_rows = sum(m['total_rows'] for m in metrics)
            comparison_data[dir_name] = {
                'anomalies': total_anomalies,
                'total_rows': total_rows,
                'percentage': (total_anomalies / total_rows * 100) if total_rows > 0 else 0
            }
        
        write_output("")
        for dir_name, data in sorted(comparison_data.items()):
            write_output(f"  {dir_name}: {data['anomalies']:,} anomalies out of {data['total_rows']:,} rows ({data['percentage']:.2f}%)")
    
    write_output("\n" + "="*70)


if __name__ == "__main__":
    extract_metrics_from_reports()
