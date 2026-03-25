import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class CreditCardFraudDetection:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        self.smote = SMOTE(random_state=42)
        
    def load_data(self, file_path='creditcard.csv'):
        """Load the credit card dataset"""
        try:
            self.data = pd.read_csv(file_path)
            print(f"Dataset loaded successfully with shape: {self.data.shape}")
            return True
        except FileNotFoundError:
            print(f"Error: {file_path} not found. Creating sample dataset for demonstration.")
            self.create_sample_dataset()
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_sample_dataset(self):
        """Create a sample dataset for demonstration purposes"""
        np.random.seed(42)
        n_samples = 10000
        n_features = 28
        
        # Generate features (V1-V28 are PCA transformed features)
        features = np.random.normal(0, 1, (n_samples, n_features))
        feature_names = [f'V{i}' for i in range(1, n_features + 1)]
        
        # Generate Time and Amount
        time = np.random.uniform(0, 172800, n_samples)  # 2 days in seconds
        amount = np.random.exponential(100, n_samples)
        
        # Create imbalanced target (99.8% non-fraud, 0.2% fraud)
        fraud_rate = 0.002
        y = np.random.choice([0, 1], n_samples, p=[1-fraud_rate, fraud_rate])
        
        # Create DataFrame
        data_dict = {'Time': time, 'Amount': amount}
        for i, name in enumerate(feature_names):
            data_dict[name] = features[:, i]
        data_dict['Class'] = y
        
        self.data = pd.DataFrame(data_dict)
        print(f"Sample dataset created with shape: {self.data.shape}")
        
    def explore_data(self):
        """Explore the dataset and display basic statistics"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print("\n=== Dataset Info ===")
        print(self.data.info())
        
        print("\n=== Basic Statistics ===")
        print(self.data.describe())
        
        print("\n=== Class Distribution ===")
        class_counts = self.data['Class'].value_counts()
        print(class_counts)
        print(f"Fraud percentage: {(class_counts[1]/len(self.data)*100):.4f}%")
        
        # Visualize class distribution
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        sns.countplot(x='Class', data=self.data)
        plt.title('Class Distribution')
        plt.xlabel('Class (0: Normal, 1: Fraud)')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        class_counts.plot(kind='pie', autopct='%1.4f%%', labels=['Normal', 'Fraud'])
        plt.title('Class Distribution Percentage')
        plt.ylabel('')
        
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def preprocess_data(self):
        """Preprocess the data for training"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return False
        
        # Separate features and target
        X = self.data.drop('Class', axis=1)
        y = self.data['Class']
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n=== Data Split ===")
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Training fraud cases: {sum(self.y_train)}")
        print(f"Test fraud cases: {sum(self.y_test)}")
        
        return True
    
    def apply_smote(self):
        """Apply SMOTE to handle class imbalance"""
        if self.X_train is None:
            print("Data not preprocessed. Please preprocess data first.")
            return False
        
        print("\n=== Applying SMOTE ===")
        print(f"Original training set shape: {self.X_train.shape}")
        print(f"Original training fraud cases: {sum(self.y_train)}")
        
        # Apply SMOTE
        self.X_train_resampled, self.y_train_resampled = self.smote.fit_resample(
            self.X_train, self.y_train
        )
        
        print(f"Resampled training set shape: {self.X_train_resampled.shape}")
        print(f"Resampled training fraud cases: {sum(self.y_train_resampled)}")
        print(f"Resampled training normal cases: {len(self.y_train_resampled) - sum(self.y_train_resampled)}")
        
        # Visualize the effect of SMOTE
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        pd.Series(self.y_train).value_counts().plot(kind='bar')
        plt.title('Original Training Set Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        pd.Series(self.y_train_resampled).value_counts().plot(kind='bar')
        plt.title('After SMOTE Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('smote_effect.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return True
    
    def train_model(self, model_type='random_forest'):
        """Train the classification model"""
        if not hasattr(self, 'X_train_resampled'):
            print("SMOTE not applied. Please apply SMOTE first.")
            return False
        
        print(f"\n=== Training {model_type.replace('_', ' ').title()} Model ===")
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1,
                class_weight='balanced'
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            )
        else:
            print("Invalid model type. Choose 'random_forest' or 'logistic_regression'")
            return False
        
        # Train the model
        self.model.fit(self.X_train_resampled, self.y_train_resampled)
        print("Model training completed successfully!")
        
        return True
    
    def evaluate_model(self):
        """Evaluate the model performance"""
        if self.model is None:
            print("No model trained. Please train a model first.")
            return False
        
        print("\n=== Model Evaluation ===")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Calculate ROC AUC score
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        print(f"ROC AUC Score: {roc_auc:.4f}")
        
        # Create confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Fraud'], 
                   yticklabels=['Normal', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance (for Random Forest)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.data.drop('Class', axis=1).columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
            plt.title('Top 10 Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
        
        return True
    
    def run_complete_pipeline(self, file_path='creditcard.csv', model_type='random_forest'):
        """Run the complete fraud detection pipeline"""
        print("=== Credit Card Fraud Detection Pipeline ===")
        
        # Load data
        if not self.load_data(file_path):
            return False
        
        # Explore data
        self.explore_data()
        
        # Preprocess data
        if not self.preprocess_data():
            return False
        
        # Apply SMOTE
        if not self.apply_smote():
            return False
        
        # Train model
        if not self.train_model(model_type):
            return False
        
        # Evaluate model
        if not self.evaluate_model():
            return False
        
        print("\n=== Pipeline Completed Successfully! ===")
        return True

def main():
    """Main function to run the credit card fraud detection"""
    # Initialize the fraud detection system
    fraud_detector = CreditCardFraudDetection()
    
    # Run the complete pipeline with user's dataset
    fraud_detector.run_complete_pipeline(file_path=r'c:\Users\VICTUS\ARPAN DOC\archive (4)\creditcard.csv', model_type='random_forest')
    
    # You can also try logistic regression
    # fraud_detector.run_complete_pipeline(file_path=r'c:\Users\VICTUS\ARPAN DOC\archive (4)\creditcard.csv', model_type='logistic_regression')

if __name__ == "__main__":
    main()
