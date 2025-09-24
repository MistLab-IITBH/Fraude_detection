from datetime import datetime
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import numpy as np # Added for robust specificity calculation

# Configuration
MODEL_PATH = '../saved_models/model.pkl'
DATA_PATH = "Output1.csv"

def evaluate_model(model_path: str, data_path: str):
    """
    Loads a pickled model and a dataset, performs predictions, and prints 
    key classification metrics including the confusion matrix.

    Args:
        model_path (str): File path to the pickled model.
        data_path (str): File path to the CSV dataset containing features and 'isFraud' target.
    """
    
    # 1. Load Model and Data
    try:
        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load the data
        data = pd.read_csv(data_path)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        return

    # 2. Prepare Data for Prediction
    y_true = data['isFraud']
    x_data = data.drop('isFraud', axis=1)
    
    # Print the head of the feature data (as in the original code)
    print(x_data.head())
    
    # 3. Predict
    y_pred = model.predict(x_data)

    # 4. Calculate Metrics
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # True Positives, False Positives, True Negatives, False Negatives
    # Note: conf_matrix is typically structured as [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = conf_matrix.ravel() # A compact way to unpack the matrix

    # Accuracy (using sklearn function)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Recall (Sensitivity) (using sklearn function)
    recall = recall_score(y_true, y_pred)
    
    # Specificity
    # Calculate denominator, ensuring no division by zero
    denominator = tn + fp
    specificity = tn / denominator if denominator != 0 else np.nan
    
    # Precision (using sklearn function)
    precision = precision_score(y_true, y_pred)
    
    # F1 Score (using sklearn function)
    f1 = f1_score(y_true, y_pred)

    # 5. Print the Results
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Execute the evaluation function
if __name__ == "__main__":
    evaluate_model(MODEL_PATH, DATA_PATH)
