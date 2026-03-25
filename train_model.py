'''
Train model on normal device behavior, then identify when devices act abnormally

Loads normal network traffic data (environment OR patient monitoring devices)
Extracts features from network packets (TCP flags, frame sizes, MQTT topics, etc.)
Converts categorical features to numbers (encoding)
Train an Isolation Forest model to learn what "normal" looks like
Save the trained model to a .pkl file for use
'''
import pandas as pd

# Choose dataset: 'environment', 'patient', or 'both'
DATASET_CHOICE = 'environment'  # Change this to test different datasets

# This could be refactored later 
def load_data(choice):

    if choice == 'environment':
        data = pd.read_csv('Dataset/environmentMonitoring_labeled.csv', low_memory=False)
        print("Loading ENVIRONMENT dataset only")
    elif choice == 'patient':
        data = pd.read_csv('Dataset/patientMonitoring_labeled.csv', low_memory=False)
        print("Loading PATIENT dataset only")
    elif choice == 'both':
        env = pd.read_csv('Dataset/environmentMonitoring_labeled.csv', low_memory=False)
        patient = pd.read_csv('Dataset/patientMonitoring_labeled.csv', low_memory=False)
        data = pd.concat([env, patient], ignore_index=True)
        print("Loading BOTH environment and patient datasets")
    else:
        raise ValueError("choice must be 'environment', 'patient', or 'both'")
    
    # Drop rows with NaN values
    data = data.dropna()
    
    print(f"Loaded {len(data)} samples")
    return data

def prepare_features():
    #split the data into features and targets
    pass

def train_models():
    # train selected models
    pass

def save_models():
    # save models 
    pass 

def main():
    print(f"Using : {DATASET_CHOICE} .csv\n")
    data = load_data(DATASET_CHOICE)

if __name__ == "__main__":
    main()