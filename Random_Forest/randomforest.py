import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib 

#Loading Dataset
dataset = pd.read_csv("../Dataset/environmentMonitoring.csv")
print("Dataset Test:")
print(dataset.head())
print(f"Shape: {dataset.shape}")

#Encoding string columns to nums so Random forest can read them
dataset['class'] = LabelEncoder().fit_transform(dataset['class'])
dataset = dataset.select_dtypes(include=['number'])

print("Dataset after converting strings to numbers:")
print(dataset.head())

#check class distribution before training
print("Label distribution:")
print(dataset['label'].value_counts())
print(dataset['label'].value_counts(normalize=True).mul(100).round(2).astype(str)+'%')

#separating features and labels
X = dataset.drop(columns=['label']) #all cols except label is feature
y = dataset['label'] #this is what we want to predict

print("Features shape:", X.shape)
print("Labels shape:", y.shape)

# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training samples:", X_train.shape[0])
print("Training label distribution:")
print(y_train.value_counts())

#training
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') #balanced => penalize misclassifying the minority class (anomalies) to learn to detect them

rf_model.fit(X_train, y_train) #fit the model on training data
y_pred = rf_model.predict(X_test) #predict on test data to check performance

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=['normal', 'anomaly']))

y_prob = rf_model.predict_proba(X_test)[:,1] #probability of anomaly class
y_pred_tuned = (y_prob >= 0.3).astype(int)

print("\nClassification report with 0.3 threshold:")
print(classification_report(y_test, y_pred_tuned, target_names=['normal', 'anomaly']))

# Save the trained model in pkl for later 
joblib.dump(rf_model, "random_forest.pkl")