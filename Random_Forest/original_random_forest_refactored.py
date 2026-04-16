import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import warnings

warnings.filterwarnings('ignore') 

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {'0x0010': 10, '0x0018': 18, '0x0012': 12, '0x0014': 14, '0x0011': 11, '0x0002': 2}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

#Loading Dataset
dataset = pd.read_csv("Dataset/environmentMonitoring_labeled.csv")

#Encoding string columns to nums so Random forest can read them
dataset = handle_non_numerical_data(dataset)

#separating features and labels
X = dataset[['tcp.dstport', 'frame.len', 'tcp.flags']] #same features as DeviceProfileTrain
y = dataset['device_label'] #this is what we want to predict - device type 

print("Features shape:", X.shape)
print("Labels shape:", y.shape)

# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training samples:", X_train.shape[0])

#training
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train) #fit the model on training data
y_pred = rf_model.predict(X_test) #predict on test data to check performance
print(classification_report(y_test, y_pred))

# Save the trained model in pkl for later 
joblib.dump(rf_model, "models/original_random_forest.pkl")

''' Explanation 
 scikit-learn showing how well your Random Forest model performed on 23 different device classes.
 Precision: Of the samples the model predicted as class X, how many were actually correct? (TP / (TP + FP))
 Recall: Of all the actual class X samples, how many did the model correctly identify? (TP / (TP + FN))
 F1-score: Harmonic mean of precision and recall — balances both metrics
 Support: Number of actual samples in the test set for that class
''' 