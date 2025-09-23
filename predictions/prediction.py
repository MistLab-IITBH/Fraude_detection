from datetime import datetime
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
model_path = '../saved_models/model.pkl'
model = pickle.load(open(model_path, 'rb'))
path = "Output1.csv"
data = pd.read_csv(path)
y_true = data['isFraud']
x_data = data.drop('isFraud', axis=1)
print(x_data.head())
y_pred = model.predict(x_data)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# True Positives, False Positives, True Negatives, False Negatives
tp = conf_matrix[1, 1]
fp = conf_matrix[0, 1]
tn = conf_matrix[0, 0]
fn = conf_matrix[1, 0]

# Accuracy
accuracy = accuracy_score(y_true, y_pred)

# Recall (Sensitivity)
recall = recall_score(y_true, y_pred)

# Specificity
specificity = tn / (tn + fp)

# Precision
precision = precision_score(y_true, y_pred)

# F1 Score
f1 = f1_score(y_true, y_pred)

# Print the results
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

