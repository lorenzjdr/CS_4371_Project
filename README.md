# IoT-anomaly-detector (IoT-AD)

## Project Summary

We are expanding the scope of this project by testing the original IoT-AD framework against real-world healthcare datasets. This includes data from IoT devices used in healthcare environments, such as patient monitoring systems. The goal is to evaluate whether the original IoT-AD framework holds up in detecting anomalies in these datasets.

Additionally, we are testing the framework against new machine learning models to determine which approach works best for identifying anomalies in the datasets. These models include Random Forest and Isolation Forest.

To facilitate this testing, we have created new datasets with injected anomalies (integrity and availability anomalies). These datasets are used to evaluate the performance of the models in detecting various types of anomalies.


refs: H. Zahan, Md W. Al Azad, I. Ali, and S. Mastorakis, "IoT-ad: A framework to detect anomalies among interconnected IoT devices." IEEE Internet of Things Journal, 2023. [Paper](https://arxiv.org/pdf/2306.06764)

Isolation Forest reference: Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou, "Isolation Forest," 2008. [Original Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)


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

## **How to Run the Project (Recommended Order)**

Below is the recommended sequence for running the main scripts in this project. This ensures that data is labeled, models are trained, anomalies are generated, and detection/rollback is performed correctly.

### 1. Label the Data
**File:** `label_data.py`

Labels the environment datasets with device information. This step is required before training or testing models.

```bash
python label_data.py
```
*Outputs labeled CSVs in the `Dataset/` folder.*

### 2. Train the Anomaly Detection Model
**File:** `train_model.py`

Trains an Isolation Forest model on the labeled environment or patient monitoring data. You can set `DATASET_CHOICE` in the script to `'environment'`, `'patient'`, or `'both'`.

```bash
python train_model.py
```
*Outputs a `.pkl` model file in the `models/` folder.*

### 3. Train a Random Forest Classifier
**File:** `Random_Forest/randomforest.py`

Trains a Random Forest classifier for comparison. Not required for the main anomaly detection pipeline.

```bash
python Random_Forest/randomforest.py
```

### 4. Generate Synthetic Anomalies
**File:** `anomaly_data/anomaly.py`

Creates synthetic integrity and availability anomalies for testing. Generates new CSVs in `anomaly_data/anomaly_datasets5/`.

```bash
python anomaly_data/anomaly.py
```

### 5. Run Detection and Rollback
**File:** `detect_and_rollback.py`

Runs the trained Isolation Forest model on attack and anomaly datasets, triggering the rollback mechanism for detected anomalies.

```bash
python detect_and_rollback.py
```

### 6. Analyze Results and Utilities
- Use `metrics/legateCSV.py` for batch anomaly prediction and CSV utilities.
- Use `device_interaction_graph.py` for graph-based analysis and visualization of device interactions.

---

**Note:** Always activate the virtual environment before running or developing the project to ensure you are using the correct dependencies.
