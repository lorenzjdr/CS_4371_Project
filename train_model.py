'''
Train model on normal device behavior, then identify when devices act abnormally

Loads normal network traffic data (environment OR patient monitoring devices)
Extracts features from network packets (TCP flags, frame sizes, MQTT topics, etc.)
Converts categorical features to numbers (encoding)
Train an Isolation Forest model to learn what "normal" looks like
Save the trained model to a .pkl file for use
'''

def load_data():
    #should load the csv file and clean it here
    pass

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
    pass

if __name__ == "__main__":
    main()