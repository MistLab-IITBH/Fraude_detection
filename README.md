# Source Code Documentation

Download "Fraud.csv" from the link: https://drive.google.com/file/d/1hZ_pLkK7uL9lQhdMmijJyWviQblAkp4p/view?usp=sharing


This folder contains the core machine learning pipeline for the Money Laundering Detection system. The source code is organized into modular components that handle different stages of the ML workflow.

## 📁 Directory Structure

```
src/
├── data_preprocessing_1.py    # Initial data cleaning and filtering
├── data_preprocessing_2.py    # Advanced data preprocessing
├── data_preprocessing_3.py    # Final data preparation
├── feature_selection.py       # Feature importance and selection
├── model_creation.py          # ML model training and evaluation
├── segmentGenerator.py        # Customer segmentation using clustering
├── catboost_info/            # CatBoost model training logs
├── logs/                     # Application logs
├── predictions/              # Model prediction outputs
├── reports/                  # Analysis reports and metrics
└── saved_models/             # Trained model artifacts
```

## 🔄 ML Pipeline Overview

The machine learning pipeline follows a structured approach:

1. **Data Preprocessing** → 2. **Segmentation** → 3. **Feature Selection** → 4. **Model Creation**

### Stage 1: Data Preprocessing (`data_preprocessing_*.py`)

#### `data_preprocessing_1.py`
- **Purpose**: Initial data loading and basic cleaning
- **Input**: Raw fraud dataset (`../Fraud.csv`)
- **Output**: Cleaned dataset with basic transformations
- **Key Functions**:
  - Data loading from CSV
  - Initial data quality checks
  - Basic data type conversions

#### `data_preprocessing_2.py` 
- **Purpose**: Advanced data cleaning and feature engineering
- **Input**: Output from stage 1
- **Output**: Enhanced dataset with engineered features
- **Key Functions**:
  - Missing value handling
  - Outlier detection and treatment
  - Feature engineering

#### `data_preprocessing_3.py`
- **Purpose**: Final data preparation for modeling
- **Input**: Output from stage 2
- **Output**: Model-ready dataset (`data_processed/filtered_data_3.csv`)
- **Key Functions**:
  - Final data validation
  - Data normalization/scaling
  - Train-test split preparation

### Stage 2: Customer Segmentation (`segmentGenerator.py`)

- **Purpose**: Cluster customers based on transaction behavior
- **Algorithm**: K-Means clustering
- **Key Features**:
  - Optimal cluster number detection
  - Silhouette score analysis
  - Customer behavior segmentation
- **Output**: 
  - Cluster assignments
  - Silhouette scores (`reports/silhoutte_scores.csv`)

### Stage 3: Feature Selection (`feature_selection.py`)

- **Purpose**: Identify most important features for fraud detection
- **Algorithm**: Random Forest feature importance
- **Input**: `data_processed/filtered_data_3.csv`
- **Output**: 
  - Feature importance rankings
  - Selected feature subset
  - Feature importance report (`reports/feature_importances.csv`)

### Stage 4: Model Creation (`model_creation.py`)

- **Purpose**: Train and evaluate the final fraud detection model
- **Algorithm**: CatBoost Classifier
- **Key Features**:
  - Model training with hyperparameter tuning
  - Performance evaluation (F1, Precision, Recall)
  - Model serialization
- **Output**:
  - Trained model (`saved_models/model.pkl`)
  - Performance metrics (`reports/performance.json`)
  - Training logs

## 🔧 Technical Stack

- **Core Libraries**:
  - `pandas` & `numpy`: Data manipulation
  - `scikit-learn`: ML algorithms and preprocessing
  - `catboost`: Gradient boosting classifier
  - `matplotlib`: Visualization

- **Key Algorithms**:
  - **CatBoost**: Main classification algorithm
  - **Random Forest**: Feature selection
  - **K-Means**: Customer segmentation

## 📊 Data Flow

```
Raw Data (Fraud.csv)
    ↓
Data Preprocessing 1 → Cleaned Data
    ↓
Data Preprocessing 2 → Enhanced Data
    ↓
Data Preprocessing 3 → Model-Ready Data
    ↓
Segment Generator → Customer Clusters
    ↓
Feature Selection → Important Features
    ↓
Model Creation → Trained Model (model.pkl)
```

## 🚀 Running the Pipeline

To execute the complete ML pipeline:

```bash

# Run the complete pipeline in sequence
python data_preprocessing_1.py
python data_preprocessing_2.py
python data_preprocessing_3.py
python segmentGenerator.py
python feature_selection.py
python model_creation.py
```

## 📋 Logging

All stages log their progress to `logs/model.txt`:
- Timestamps for each operation
- Processing status updates
- Error handling information
- Performance metrics

## 📈 Output Files

### Reports (`reports/`)
- `feature_importances.csv`: Feature ranking by importance
- `missing_values.csv`: Data quality analysis
- `performance.json`: Model evaluation metrics
- `silhoutte_scores.csv`: Clustering quality metrics

### Models (`saved_models/`)
- `model.pkl`: Serialized trained CatBoost model

### Predictions (`predictions/`)
- Various output CSV files with fraud predictions
- `prediction.py`: Prediction utility script

## 🔍 Model Performance

The CatBoost model is optimized for:
- **High Recall**: Minimize false negatives (missed fraud cases)
- **Balanced Precision**: Reduce false positives for operational efficiency
- **F1 Score**: Overall classification performance


