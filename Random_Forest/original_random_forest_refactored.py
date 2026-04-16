import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib 

#Loading Dataset
dataset = pd.read_csv("Dataset/environmentMonitoring_labeled.csv")
print("Dataset Test:")
print(dataset.head())
print(f"Shape: {dataset.shape}")

#Encoding string columns to nums so Random forest can read them
dataset['class'] = LabelEncoder().fit_transform(dataset['class'])
device_label_encoder = LabelEncoder()
dataset['device_label'] = device_label_encoder.fit_transform(dataset['device_label'].fillna('Unknown'))
dataset = dataset.select_dtypes(include=['number'])

print("Dataset after converting strings to numbers:")
print(dataset.head())

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