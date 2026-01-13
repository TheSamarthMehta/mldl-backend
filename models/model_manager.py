"""
Model Manager Module
Handles model loading, predictions, and model lifecycle
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages machine learning models
    """
    
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_type = None
        self.model_path = model_path or os.path.join('models', 'trained', 'cardio_model.pkl')
        self.model_info = {}
        self.load_default_model()
    
    def load_default_model(self):
        """Load the default model if it exists"""
        try:
            # First check if specific model path exists
            if os.path.exists(self.model_path):
                self.load_model(self.model_path)
                logger.info(f"Default model loaded from {self.model_path}")
            else:
                # Try to find any trained model in the trained directory
                trained_dir = os.path.join('models', 'trained')
                if os.path.exists(trained_dir):
                    pkl_files = [f for f in os.listdir(trained_dir) if f.endswith('.pkl')]
                    if pkl_files:
                        # Load the most recent model
                        latest_model = max(pkl_files, key=lambda f: os.path.getmtime(os.path.join(trained_dir, f)))
                        model_path = os.path.join(trained_dir, latest_model)
                        self.load_model(model_path)
                        logger.info(f"Loaded latest model: {latest_model}")
                    else:
                        logger.warning(f"No trained models found in {trained_dir}")
                else:
                    logger.warning(f"No default model found at {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading default model: {str(e)}")
    
    def load_model(self, model_path):
        """
        Load a trained model from disk
        
        Args:
            model_path: Path to the saved model file
        """
        try:
            model_data = joblib.load(model_path)
            
            # Handle both old format (just model) and new format (dict with model and scaler)
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.feature_names = model_data.get('feature_names')
                self.model_type = model_data.get('model_type')
            else:
                # Old format - just the model
                self.model = model_data
                self.scaler = None
                self.feature_names = None
                self.model_type = None
            
            self.model_path = model_path
            self.model_info = {
                'path': model_path,
                'loaded_at': datetime.now().isoformat(),
                'type': type(self.model).__name__,
                'has_scaler': self.scaler is not None,
                'feature_names': self.feature_names
            }
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def save_model(self, model, model_path=None):
        """
        Save a trained model to disk
        
        Args:
            model: The trained model object
            model_path: Path to save the model
        """
        try:
            save_path = model_path or self.model_path
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            joblib.dump(model, save_path)
            self.model = model
            self.model_path = save_path
            
            logger.info(f"Model saved successfully to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def predict(self, input_data):
        """
        Make predictions using the loaded model
        
        Args:
            input_data: Input data for prediction (dict or DataFrame)
            
        Returns:
            Prediction results
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        try:
            # Convert input to DataFrame if it's a dict
            if isinstance(input_data, dict):
                # Handle single prediction
                if 'features' in input_data:
                    features = input_data['features']
                    df = pd.DataFrame([features])
                else:
                    df = pd.DataFrame([input_data])
            else:
                df = pd.DataFrame(input_data)
            
            # Apply feature engineering for cardiovascular data
            if 'age' in df.columns and 'age_years' not in df.columns:
                # Convert age from days to years if needed
                df['age_years'] = df['age'] / 365.25
                df = df.drop('age', axis=1)
            
            # Calculate BMI if not present
            if 'bmi' not in df.columns and 'height' in df.columns and 'weight' in df.columns:
                df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
            
            # Calculate pulse pressure if not present
            if 'pulse_pressure' not in df.columns and 'ap_hi' in df.columns and 'ap_lo' in df.columns:
                df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
            
            # Ensure features are in correct order
            if self.feature_names:
                # Make sure all required features are present
                missing_features = set(self.feature_names) - set(df.columns)
                if missing_features:
                    raise ValueError(f"Missing features: {missing_features}")
                # Reorder columns to match training data
                df = df[self.feature_names]
            
            # Apply scaling if scaler is available
            if self.scaler is not None:
                df_scaled = self.scaler.transform(df)
                df = pd.DataFrame(df_scaled, columns=df.columns)
            
            # Make prediction
            prediction = self.model.predict(df)
            
            # Get probability if available
            result = {
                'prediction': int(prediction[0]) if len(prediction) == 1 else prediction.tolist(),
                'risk_level': 'High Risk' if prediction[0] == 1 else 'Low Risk'
            }
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(df)
                result['probability'] = float(probabilities[0][1]) if len(probabilities) == 1 else probabilities.tolist()
                result['confidence'] = float(max(probabilities[0]))
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def evaluate(self, test_data):
        """
        Evaluate model performance
        
        Args:
            test_data: Test dataset with features and labels
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        try:
            X_test = test_data.get('features')
            y_test = test_data.get('labels')
            
            if X_test is None or y_test is None:
                raise ValueError("Test data must contain 'features' and 'labels'")
            
            # Convert to DataFrame
            X_test_df = pd.DataFrame(X_test)
            
            # Get predictions
            predictions = self.model.predict(X_test_df)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'accuracy': float(accuracy_score(y_test, predictions)),
                'precision': float(precision_score(y_test, predictions, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, predictions, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_test, predictions, average='weighted', zero_division=0))
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            raise
    
    def is_model_loaded(self):
        """Check if a model is currently loaded"""
        return self.model is not None
    
    def get_model_info(self):
        """Get information about the current model"""
        if self.model is None:
            return {'status': 'No model loaded'}
        
        info = self.model_info.copy()
        
        # Add model parameters if available
        if hasattr(self.model, 'get_params'):
            info['parameters'] = self.model.get_params()
        
        return info
    
    def get_current_model(self):
        """Get the name of the current model"""
        if self.model is None:
            return None
        return os.path.basename(self.model_path)
    
    def get_available_models(self):
        """Get list of available trained models"""
        models_dir = os.path.join('models', 'trained')
        
        if not os.path.exists(models_dir):
            return []
        
        models = []
        for file in os.listdir(models_dir):
            if file.endswith('.pkl') or file.endswith('.joblib'):
                models.append(file)
        
        return models
