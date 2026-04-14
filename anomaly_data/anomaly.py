import pandas as pd
import random
import os

DATASET = "../Dataset/environmentMonitoring.csv" #og dataset
OUTPUT_DATA = "anomaly_datasets50"   #anomaly log
INTEGRITY_COUNT = 50
AVAILABILITY_COUNT = 50

# columns that can be corrupted
Integrity_Columns=[
    "frame.len", "tcp.len", "tcp.window_size_value",
    "tcp.checksum", "mqtt.kalive"
]


os.makedirs(OUTPUT_DATA, exist_ok=True) #create output folder
df = pd.read_csv(DATASET) #read dataset

df_integrity = df.copy()
df_availability = df.copy()

#anomaly log
log_columns = ["dataset_name", "anomaly_type", "row_index", "column", "old_value", "new_value"]
log = []

#INTEGRITY ANOMALY CREATION
rows_to_corrupt = random.sample(range(len(df_integrity)), INTEGRITY_COUNT)

for i in rows_to_corrupt:
    col = random.choice(Integrity_Columns)
    old_val = df_integrity.loc[i, col]

    if pd.api.types.is_numeric_dtype(df_integrity[col]):
        new_val = old_val + random.randint(1,1000)
    else:
        new_val = str(old_val) + "_CORRUPTED"

    df_integrity.loc[i, col]= new_val
    df_integrity.loc[i, 'label'] =1 #mark as anomaly
    log.append(["integrity_dataset.csv", "integrity", i, col, old_val, new_val])

#save integrity anomaly dataset
integrity_file = os.path.join(OUTPUT_DATA, "integrity_dataset50.csv")
df_integrity.to_csv(integrity_file, index=False)

#AVAILABILITY ANOMALY CREATION
rows_to_delete = random.sample(range(len(df_availability)), AVAILABILITY_COUNT)

df_availability.loc[rows_to_delete, 'label']=1 #mark before dropping

for i in rows_to_delete:
    log.append(["availability_dataset.csv", "availability", i, "row_deleted", "row_exists", "row_removed"])

#df_availability.drop(index=rows_to_delete, inplace=True)
availability_file = os.path.join(OUTPUT_DATA, "availability_dataset50.csv")
df_availability.to_csv(availability_file, index=False)

#SAVING
log_df = pd.DataFrame(log, columns=log_columns)
log_file = os.path.join(OUTPUT_DATA, "anomaly_log50.csv")
log_df.to_csv(log_file, index=False)

print("Datasets created and logged successfully!")
print(f"Integrity anomalies: {integrity_file}")
print(f"Availability anomalies: {availability_file}")
print(f"Log file: {log_file}")