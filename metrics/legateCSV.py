"""
CS_4371_Project — CSV Dataset Reader Utilities

Purpose
-------
Centralized helpers to load and validate the project's CSV datasets. Import
functions from this module instead of calling `pandas.read_csv` directly so
that file paths, NA handling, encodings, and date parsing stay consistent
across the codebase.

Responsibilities
---------------
- Locate CSV files under `Dataset/` and `anomaly_data/` subdirectories.
- Read data with common defaults (UTF‑8, standard NA values, safe dtypes).
- Optionally enforce/validate expected columns and parse timestamps.
- Offer chunked reads for large files to reduce memory pressure.

Intended usage (examples)
-------------------------
from metrics.legateCSV import read_dataset  # to be implemented below

# Example: load labeled environment monitoring data
df = read_dataset(
    "Dataset/environmentMonitoring_labeled.csv",
    parse_dates=["timestamp"],
)

# Example: iterate over a large anomaly log in chunks
for chunk in read_dataset(
    "anomaly_data/anomaly_datasets5/anomaly_log5.csv", chunksize=100_000
):
    process(chunk)
"""

#Imports
import pandas as pd
import numpy as np
import pathlib

#Constants
root_dir = pathlib.Path('../anomaly_data/')
file_pattern_integrity = "anomaly_datasets*/integrity_dataset*.csv"
file_pattern_availability = "anomaly_datasets*/availability_dataset*.csv"

def find_files(pattern : str):
    for file_path in root_dir.glob(pattern):
        print(f"Reading file: {file_path.name} from {file_path.parent}")



#REMOVE AFTER TESTING
if __name__ == '__main__':
    find_files(file_pattern_integrity)