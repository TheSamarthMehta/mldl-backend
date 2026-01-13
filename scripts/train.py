"""
Model Training Script
Handles model training workflows
"""

import os
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_data, preprocess_data, split_data
from utils.validators import validate_training_data
from models.model_manager import ModelManager

# ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)


def get_model(model_type='logistic_regression'):
    """
    Get model instance based on type
    
    Args:
        model_type: Type of model to create
    
    Returns:
        Model instance
    """
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'svm': SVC(kernel='rbf', probability=True, random_state=42),
        'default': LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    }
    
    return models.get(model_type, models['default'])


def train_model(model_type='logistic_regression', data_path=None):
    """
    Train a machine learning model for cardiovascular disease prediction
    
    Args:
        model_type: Type of model to train
        data_path: Path to training data (optional)
    
    Returns:
        Dictionary with training results
    """
    try:
        logger.info(f"Starting training for {model_type} model")
        
        # Load data
        if data_path is None:
            data_path = os.path.join('data', 'cardio_train.csv')
        
        if not os.path.exists(data_path):
            logger.error(f"Data file not found at {data_path}")
            return {
                'success': False,
                'error': f'Data file not found at {data_path}. Please ensure cardio_train.csv exists in the data folder.'
            }
        
        # Load and preprocess data
        logger.info("Loading cardiovascular disease data...")
        import pandas as pd
        # Load with semicolon delimiter
        df = pd.read_csv(data_path, sep=';')
        
        logger.info(f"Data loaded: {len(df)} samples, {len(df.columns)} columns")
        
        # Drop id column if exists
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        # Feature engineering
        # Convert age from days to years
        if 'age' in df.columns:
            df['age_years'] = df['age'] / 365.25
            df = df.drop('age', axis=1)
        
        # Calculate BMI
        if 'height' in df.columns and 'weight' in df.columns:
            df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        
        # Calculate pulse pressure
        if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
            df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
        
        # Target column
        target_column = 'cardio'
        
        logger.info(f"Target column: {target_column}")
        logger.info(f"Features: {[col for col in df.columns if col != target_column]}")
        
        # Remove outliers
        df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 220)]
        df = df[(df['ap_lo'] >= 50) & (df['ap_lo'] <= 140)]
        df = df[(df['height'] >= 140) & (df['height'] <= 210)]
        df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]
        
        logger.info(f"After outlier removal: {len(df)} samples")
        
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Validate
        is_valid, error_msg = validate_training_data(X, y)
        if not is_valid:
            raise ValueError(f"Data validation failed: {error_msg}")
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
        
        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve column names
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Get model
        model = get_model(model_type)
        logger.info(f"Training {model.__class__.__name__}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        precision = precision_score(y_test, test_predictions)
        recall = recall_score(y_test, test_predictions)
        f1 = f1_score(y_test, test_predictions)
        
        # ROC AUC
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = None
        
        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        if roc_auc:
            logger.info(f"ROC AUC: {roc_auc:.4f}")
        
        # Save model and scaler together
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"cardio_model_{model_type}_{timestamp}.pkl"
        model_path = os.path.join('models', 'trained', model_filename)
        
        # Save both model and scaler
        import joblib
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': list(X_train.columns),
            'model_type': model_type
        }
        joblib.dump(model_data, model_path)
        logger.info(f"Model and scaler saved to {model_path}")
        
        # Return results
        metrics = {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': list(X_train.columns)
        }
        
        if roc_auc:
            metrics['roc_auc'] = float(roc_auc)
        
        return {
            'success': True,
            'model_path': model_path,
            'model_type': model_type,
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Train model with Logistic Regression (default)
    result = train_model('logistic_regression')
    
    if result['success']:
        print("Training completed successfully!")
        print(f"Model saved to: {result['model_path']}")
        print(f"Metrics: {result['metrics']}")
    else:
        print(f"Training failed: {result['error']}")
