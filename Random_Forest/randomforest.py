import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Loading Dataset
dataset = pd.read_csv("../Dataset/environmentMonitoring.csv")
print("Dataset Test:")
print(dataset.head())
print(f"Shape: {dataset.shape}")

#Encoding string columns to nums so Random forest can read them
srting_cols = ['class', 'device_label']
for col in srting_cols:
    dataset[col] = LabelEncoder().fit_transform(dataset[col])

# Drop string columns that won't help 
drop_cols = ['ip.src','ip.dst','tcp.flags','tcp.payload','mqtt.clientid','mqtt.msg','mqtt.topic']
dataset = dataset.drop(columns=drop_cols) 

print("Dataset after converting strings to numbers:")
print(dataset.head())