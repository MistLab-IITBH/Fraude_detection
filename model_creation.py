import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
import logging
import os # Added for directory creation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from catboost import CatBoostClassifier

# --- Configuration ---
LOG_FILE = 'logs/model.txt'
DATA_PATH = "data_processed/final_data.csv"
REPORTS_DIR = "reports"
MODEL_DIR = "saved_models"
PERFORMANCE_FILE = f"{REPORTS_DIR}/performance.json"
MISSING_VALUES_FILE = f"{REPORTS_DIR}/missing_values.csv"
MODEL_PATH = f"{MODEL_DIR}/model.pkl"

# --- Setup Directories and Logging ---
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(filename=LOG_FILE,
                    filemode='a',
                    format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

logging.warning("----------")
logging.warning("MODEL CREATION STAGE")

# --- Step 1: Data Loading ---
logging.warning("Reading Final Dataset...")

# Read the CSV directly into a DataFrame
dataMat = pd.read_csv(DATA_PATH)

logging.warning("Read Final Dataset")

# --- Step 2: Feature Analysis and Missing Values ---
logging.warning("Checking Categorical Features...")

# Identify categorical features based on object dtype
cat_feat = dataMat.select_dtypes(include='object').columns.tolist()

logging.warning("Checking Missing Values...")

# Use pandas to get missing values count for all columns
missing = dataMat.isnull().sum().reset_index()
missing.columns = ['features', 'null_values_count']

# Filter out features with zero missing values for cleaner logging
missing = missing[missing['null_values_count'] > 0]

logging.warning("Storing Missing Values...")

# Store the missing values report
missing.to_csv(MISSING_VALUES_FILE, index=False)

logging.warning("Storing Missing Values Done")

# --- Step 3: Encoding Categorical Features ---
logging.warning("Encoding Categorical Features...")

encoder = LabelEncoder()
for col in cat_feat:
    # Use .loc to ensure modification of the DataFrame slice
    dataMat.loc[:, col] = encoder.fit_transform(dataMat[col])

logging.warning("Features Encoding Done")

# --- Step 4: Data Splitting ---
logging.warning("Creating X and y variables ...")

# Use .drop() with axis=1 for X and direct column selection for y
X = dataMat.drop(columns=['isFraud'])
y = dataMat['isFraud']

logging.warning(f"Shape of X: {X.shape} and Shape of y: {y.shape}")

logging.warning("Splitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Step 5: Model Training ---
logging.warning("Instantiating Model...")

# CatBoostClassifier with specified parameters
model = CatBoostClassifier(random_state=42, class_weights={0:1, 1:12}, silent=True)

logging.warning("Fitting Model...")

model.fit(X_train, y_train)

# Prediction on the test set
y_pred_cat = model.predict(X_test)

# --- Step 6: Model Saving ---
logging.warning("Saving Model...")

# Save the trained model using pickle
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

logging.warning("Saving Model Metrics...")

# --- Step 7: Metrics Calculation and Storage ---

# Load existing metrics file (assuming it's initialized correctly in a previous run)
try:
    with open(PERFORMANCE_FILE, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    # If file doesn't exist, initialize the structure
    data = {"model_metric": []}
except json.JSONDecodeError:
    # Handle empty or malformed JSON file
    data = {"model_metric": []}


# Calculate metrics
model_metric = {
    "time_stamp": datetime.now().strftime("%d-%m-%Y_%H:%M:%S"),
    "confusion_matrix": confusion_matrix(y_test, y_pred_cat).tolist(),
    "precision": precision_score(y_test, y_pred_cat),
    "recall": recall_score(y_test, y_pred_cat),
    "f1_score": f1_score(y_test, y_pred_cat)
}

# Append and save the updated metrics
data['model_metric'].append(model_metric)
with open(PERFORMANCE_FILE, "w") as f:
    json.dump(data, f, indent=4)

logging.warning("Model Metrics Stored")
