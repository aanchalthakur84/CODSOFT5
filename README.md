# Credit Card Fraud Detection

A comprehensive credit card fraud detection system that handles severe class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

## Features

- **Data Exploration**: Comprehensive analysis of the credit card dataset
- **Class Imbalance Handling**: Uses SMOTE to balance the dataset
- **Multiple Models**: Supports Random Forest and Logistic Regression
- **Comprehensive Evaluation**: ROC curves, confusion matrices, and classification reports
- **Feature Importance**: Identifies the most important features for fraud detection
- **Visualization**: Generates various plots for better understanding

## Installation

Install the required dependencies:

```bash
pip install imbalanced-learn scikit-learn pandas numpy matplotlib seaborn
```

## Usage

### Basic Usage

```python
from credit_card_fraud_detection import CreditCardFraudDetection

# Initialize the detector
detector = CreditCardFraudDetection()

# Run the complete pipeline
detector.run_complete_pipeline(file_path='creditcard.csv', model_type='random_forest')
```

### Step-by-Step Usage

```python
# Initialize
detector = CreditCardFraudDetection()

# Load data (will create sample data if file not found)
detector.load_data('creditcard.csv')

# Explore the dataset
detector.explore_data()

# Preprocess the data
detector.preprocess_data()

# Apply SMOTE to handle class imbalance
detector.apply_smote()

# Train the model
detector.train_model(model_type='random_forest')

# Evaluate the model
detector.evaluate_model()
```

## Dataset

The system expects a credit card dataset with the following structure:
- `Time`: Time elapsed between transactions
- `V1-V28`: PCA-transformed features (anonymized)
- `Amount`: Transaction amount
- `Class`: Target variable (0: Normal, 1: Fraud)

If no dataset is provided, the system will generate a sample dataset for demonstration purposes.

## Model Performance

The system evaluates models using:
- **Classification Report**: Precision, Recall, F1-Score
- **ROC AUC Score**: Area under the ROC curve
- **Confusion Matrix**: True/False positives and negatives
- **Feature Importance**: Most predictive features (for Random Forest)

## Output Files

The system generates the following visualization files:
- `class_distribution.png`: Shows the imbalance in the dataset
- `smote_effect.png`: Visualizes the effect of SMOTE
- `confusion_matrix.png`: Confusion matrix heatmap
- `roc_curve.png`: ROC curve plot
- `feature_importance.png`: Top 10 important features

## Class Imbalance Handling

Credit card fraud detection typically suffers from severe class imbalance (fraud cases are rare). This system uses:

1. **SMOTE**: Creates synthetic samples of the minority class
2. **Stratified Sampling**: Ensures balanced representation in train/test splits
3. **Class Weighting**: Additional balancing during model training

## Supported Models

- **Random Forest**: Ensemble method with feature importance
- **Logistic Regression**: Linear model with regularization

## Example Output

```
=== Dataset Info ===
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 31 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Time    10000 non-null  float64
 1   Amount  10000 non-null  float64
 2   V1      10000 non-null  float64
 ...
 29  V28     10000 non-null  float64
 30  Class   10000 non-null  int64  

=== Class Distribution ===
0    9980
1      20
Name: Class, dtype: int64
Fraud percentage: 0.2000%

=== Classification Report ===
              precision    recall  f1-score   support
           0       1.00      1.00      1.00      1996
           1       0.50      0.50      0.50         4

ROC AUC Score: 0.8750
```

## Notes

- The system automatically creates a sample dataset if `creditcard.csv` is not found
- SMOTE is applied only to the training data to prevent data leakage
- All visualizations are saved as PNG files for easy sharing
- The system handles both binary classification and provides comprehensive metrics

## Requirements

- Python 3.7+
- imbalanced-learn
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
