import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

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

#separating features and labels
X = dataset.drop(columns=['label']) #all cols except label is feature
y = dataset['label'] #this is what we want to predict

print("Features shape:", X.shape)
print("Labels shape:", y.shape)

# split train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])