import pandas as pd # Use pd alias for convention
import numpy as np
import json
import csv
import logging
from random import randint
import os # Added for robust directory creation

# --- Configuration ---
data_path = "data_processed/filtered_data.csv"
log_file = 'logs/model.txt'

# --- Setup Logging ---
# Ensure logs directory exists (optional, but good practice)
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file,
                    filemode='a',
                    format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

logging.warning("----------")
logging.warning("DATA PREPROCESSING 2 STAGE")

# --- Step 1: Read Data ---
logging.warning("Reading Preprocessed 1 dataset...")

# Read the CSV directly into a DataFrame
df = pd.read_csv(data_path)

logging.warning("Read Preprocessed 1 dataset...")

# The original column names from the raw data are used here for mapping.
# Based on the structure, the columns are likely:
# ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
#  'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud']

# The original code's column indices were based on a NumPy array, which is inefficient.
# We'll rely on the existing column names from the CSV.
# Since the previous script didn't write headers, we must assume the order.
# Based on the previous script's filtering, the column names (before change) should be:
df.columns = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
              'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud']

# Drop 'newbalanceOrig' and 'newbalanceDest' to match the 10 columns in the original output list
df = df.drop(columns=['newbalanceOrig', 'newbalanceDest'], errors='ignore')


# --- Step 2: Change Labels of 'type' Column (Vectorized) ---
logging.warning("Changing Labels of Type Column ...")

# 1. Map 'PAYMENT' to 'CREDIT'
df['type'] = df['type'].replace('PAYMENT', 'CREDIT')

# 2. Map 'TRANSFER' to random 'WIRE_IN' or 'WIRE_OUT'
transfer_mask = (df['type'] == 'TRANSFER')
num_transfers = transfer_mask.sum()
if num_transfers > 0:
    # Generate a list of random choices: 0 (WIRE_IN) or 1 (WIRE_OUT)
    # This maintains the exact behavior of the original random choice in the loop.
    random_wires = [('WIRE_IN', 'WIRE_OUT')[randint(0, 1)] for _ in range(num_transfers)]
    df.loc[transfer_mask, 'type'] = random_wires

logging.warning("Changing Labels Done")


# --- Step 3: Create 'accountType' Column (Vectorized) ---
# Note: In the original code, the 'type' column was modified *before* checking it for 'TRANSFER'
# to set 'accountType'. The new approach uses the *original* 'type' column for this check
# for clarity and efficiency, which yields the same result because only 'TRANSFER' is relevant.

logging.warning("Creating Account Type Column ...")

# Create a new column 'accountType' based on the *original* 'type' values
# 'TRANSFER' is mapped to 'FOREIGN', everything else is 'DOMESTIC'
df['accountType'] = np.where(df['type'].isin(['WIRE_IN', 'WIRE_OUT']), 'FOREIGN', 'DOMESTIC')

# Rename the 'type' column to 'trans_type' as per the desired output
df = df.rename(columns={'type': 'trans_type'})

logging.warning("Account Type Column Created")


# --- Step 4: Reorder and Save Data ---
logging.warning("Storing Data in Data_processed Folder...")

# Define the final column order as specified in the original code
columns = ['step', 'trans_type', 'amount', 'nameOrig', 'oldbalanceOrg',
           'nameDest', 'oldbalanceDest', 'accountType', 'isFraud', 'isFlaggedFraud']

# Reorder the columns and save the final DataFrame
data_primary = df[columns]
data_primary.to_csv('data_processed/filtered_data_2.csv', index=False)

logging.warning("Storing Data Done")
