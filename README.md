# IoT-anomaly-detector (IoT-AD)

This is the code used for the following publication: H. Zahan, Md W. Al Azad, I. Ali, and S. Mastorakis, "IoT-ad: A framework to detect anomalies among interconnected IoT devices." IEEE Internet of Things Journal, 2023.

This GitHub repository contains a collection of scripts for analyzing device actions in a smart home environment. The scripts are designed to preprocess data, train models, detect anomalies, and rollback device actions if necessary. The following is an overview of the scripts included in this project: 

**Scripts:** 

**1. DeviceStateData.py:** This script controls the device actions in a smart home by monitoring and managing the state of devices.  

**2. Preprocess.py:** This script converts pcap files to CSV format, making it easier to analyze the data and extract meaningful information.  

**3. DeviceProfileTrain.py:** In this script, the machine learning model is trained using device action data, and a signature is created for each device's action pattern.  

**4. trainandtest.py:** This script focuses on training the machine learning model using multiple device actions and their interactions. It also includes testing and evaluation of the trained model. 

**5. complexmodel.py:** This script builds a deep learning model that can handle complex interactions and multiple device actions. It trains the model using appropriate data and fine-tunes it for optimal performance.  

**6. validation.py:** Using the trained model, this script performs anomaly detection on device actions. It identifies any unusual or unexpected behavior and flags them as anomalies.  

**7. Rollback.py:** In the event of detecting interaction anomalies, this script rolls back the device actions to their stable state, ensuring the integrity and security of the smart home environment. 

---

## **New Additions:**

We are expanding the scope of this project by testing the original IoT-AD framework against real-world healthcare datasets. This includes data from IoT devices used in healthcare environments, such as patient monitoring systems. The goal is to evaluate whether the original IoT-AD framework holds up in detecting anomalies in these datasets.

Additionally, we are testing the framework against new machine learning models to determine which approach works best for identifying anomalies in the datasets. These models include Random Forest and Isolation Forest.

To facilitate this testing, we have created new datasets with injected anomalies (integrity and availability anomalies). These datasets are used to evaluate the performance of the models in detecting various types of anomalies.

---

## **Getting Started:**  

To use these scripts, follow these steps:  

1. Clone the repository to your local machine.  

2. Install the necessary dependencies and libraries required for running the scripts.  

3. Run each script in the specified sequence as described above, ensuring the proper data inputs and configuration settings.

---

## **Setting Up a Python Virtual Environment**

To keep your dependencies organized and avoid conflicts, it is recommended to use a Python virtual environment.

### 1. Create a Virtual Environment

Open your terminal or command prompt and navigate to the project directory. Run:

```bash
python -m venv venv
```

This will create a folder named `venv` in your project directory.

### 2. Activate the Virtual Environment

- **On Windows:**
	```bash
	venv\Scripts\activate
	```
- **On macOS/Linux:**
	```bash
	source venv/bin/activate
	```

After activation, your terminal prompt should show `(venv)`.

### 3. Install Project Requirements

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Deactivate the Virtual Environment

When finished, you can deactivate the environment by running:

```bash
deactivate
```

---

**Note:** Always activate the virtual environment before running or developing the project to ensure you are using the correct dependencies.
