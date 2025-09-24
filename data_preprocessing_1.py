import pandas as pd
import numpy as np
import json
import csv
import logging
import os # Added for robust directory creation
from random import randint

# --- Configuration ---
data_path = "../Fraud.csv"
output_dir = "data_processed"
log_file = 'logs/model.txt'
MIN_TRANSACTION_COUNT = 40 # Threshold for filtering

# --- Setup Logging and Output Directory ---
os.makedirs(os.path.dirname(log_file), exist_ok=True) # Ensure logs directory exists
logging.basicConfig(filename=log_file,
                    filemode='a',
                    format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

logging.warning("DATA PREPROCESSING 1 STAGE")

# --- Step 1: Read and Prepare Data (Using Pandas for efficiency) ---
logging.warning("Reading Dataset...")

# Use pandas to read the CSV
df = pd.read_csv(data_path)

logging.warning("Read Dataset")

# --- Step 2: Calculate Transaction Counts (Vectorized) ---
logging.warning("Checking Each Person's Transactions Count...")

# Combine both columns and calculate counts once
# This avoids the slow, manual iteration loop and dictionary management.
name_columns = ['nameOrig', 'nameDest']
all_names = pd.concat([df['nameOrig'], df['nameDest']])
name_counts = all_names.value_counts()

logging.warning("Count Identification Done")

# --- Step 3: Calculate Median (Optimized) ---
logging.warning("Calculating Median ...")

# Filter counts based on the threshold (value > 40)
countArr = name_counts[name_counts > MIN_TRANSACTION_COUNT].values
median = np.median(countArr)

logging.warning(f"Median : {median}")

# --- Step 4: Filter Data (Vectorized) ---
logging.warning("Filtering Data Based on Transactions Count...")

# Create boolean masks to identify rows where nameOrig count > 40 OR nameDest count > 40
orig_mask = df['nameOrig'].map(name_counts) > MIN_TRANSACTION_COUNT
dest_mask = df['nameDest'].map(name_counts) > MIN_TRANSACTION_COUNT

# Combine masks with logical OR
filtered_df = df[orig_mask | dest_mask]

logging.warning("Filtering Done")

# Convert the filtered DataFrame to a list of lists (NumPy array required by original logic)
# Note: The original code used X = X.to_numpy() early on, but here we use the DataFrame for efficiency
# and only convert to list-of-lists (which is what writerows expects) at the end.
csv_golden_data = filtered_df.values.tolist() # Convert DataFrame values to list of lists

# --- Step 5: Store Filtered Data ---
logging.warning("Storing Filtered Data in data_processed folder...")

# Ensure the output directory exists (crucial fix from previous errors)
os.makedirs(output_dir, exist_ok=True)

with open(f"{output_dir}/filtered_data.csv", "w") as f:
    writer = csv.writer(f)
    # The original script does not write headers; we maintain that behavior.
    writer.writerows(csv_golden_data)

logging.warning("Data is Stored")
