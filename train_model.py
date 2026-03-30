'''
Train model on normal device behavior, then identify when devices act abnormally

Loads normal network traffic data (environment OR patient monitoring devices)
Extracts features from network packets (TCP flags, frame sizes, MQTT topics, etc.)
Converts categorical features to numbers (encoding)
Train an Isolation Forest model to learn what "normal" looks like
Save the trained model to a .pkl file for use
'''
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import pickle

# Choose dataset: 'environment', 'patient', or 'both'
DATASET_CHOICE = 'environment'  # Change this to test different datasets

# This could be refactored later 
def load_data(choice):

    if choice == 'environment':
        data = pd.read_csv('Dataset/environmentMonitoring.csv', low_memory=False)
        print("Loading ENVIRONMENT dataset only")
    elif choice == 'patient':
        data = pd.read_csv('Dataset/patientMonitoring.csv', low_memory=False)
        print("Loading PATIENT dataset only")
    elif choice == 'both':
        env = pd.read_csv('Dataset/environmentMonitoring.csv', low_memory=False)
        patient = pd.read_csv('Dataset/patientMonitoring.csv', low_memory=False)
        data = pd.concat([env, patient], ignore_index=True)
        print("Loading BOTH environment and patient datasets")
    else:
        raise ValueError("choice must be 'environment', 'patient', or 'both'")
    
    # Drop rows with NaN values
    data = data.dropna()
    
    print(f"Loaded {len(data)} samples")
    return data

def prepare_features(data):
    """Encode categorical features and return X"""
    # Drop label columns and non-feature columns
    X = data.drop(['class', 'label', 'device_label'], axis=1, errors='ignore').copy()
    
    # Encode categorical columns
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Convert all columns to numeric and handle any remaining objects
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Ensure all columns are numeric
    X = X.astype(float)
    
    print(f"Features shape: {X.shape}")
    return X, label_encoders

def train_isolation(X):
    """Train Isolation Forest anomaly detector"""
    # mark .001% of the data as anomalies. So the model knows what "normal" looks like
    # assumes that the data set has no anomalies
    model = IsolationForest(contamination=0.001, random_state=42, n_jobs=-1)
    model.fit(X)
    print("Isolation Forest trained")    
    return model    

def save_model(model, encoders, dataset_choice, model_name):
    """Save model and encoders"""
    
    with open(model_name + "_" + dataset_choice + '.pkl', 'wb') as f:
        pickle.dump({
            'model': model, 
            'encoders': encoders,
            'dataset': dataset_choice
        }, f)
    
    print(f"Model saved to model.pkl (trained on: {dataset_choice})")

def main():
    print(f"Using : {DATASET_CHOICE} .csv\n")
    data = load_data(DATASET_CHOICE)
    X, encoders = prepare_features(data)
    isolation_forest_model = train_isolation(X)
    save_model(isolation_forest_model, encoders, DATASET_CHOICE,"isolation_forest_model")

if __name__ == "__main__":
    main()