import time
import os
import pickle
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
import seaborn as sn
from sklearn.metrics import confusion_matrix

dataset_files = [
    '../anomaly_data/anomaly_datasets3/integrity_dataset3.csv',
    '../anomaly_data/anomaly_datasets5/integrity_dataset5.csv',
    '../anomaly_data/anomaly_datasets50/integrity_dataset50.csv',
]

frames = [pd.read_csv(f) for f in dataset_files]
df = pd.concat(frames, ignore_index=True)
print(df.head())
df = df.drop(columns=['ip.src', 'ip.dst'], errors='ignore')
df = df.dropna()

def handle_non_numerical_data(df):
    columns = df.columns.values
    encoders = {}
    for column in columns:
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            encoders[column] = le
    return df, encoders

df, encoders = handle_non_numerical_data(df)
start_time = time.time()

#Model Train
data = shuffle(df)
X = data.drop(columns=['label'])  # all columns except label are features
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)

models = {}

#Random Forest

models['Random Forest'] = RandomForestClassifier()

accuracy, precision, recall = {}, {}, {}
for key in models.keys():
    print(key)
    # Fit the classifier model
    pkl_name = key.replace(" ", '') + '_' + 'OGCode' + '.pkl'
    trained_model = models[key].fit(X_train, y_train)
    joblib.dump({'model': trained_model, 'encoders': encoders}, '../models/' + pkl_name)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
