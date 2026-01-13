"""
Standalone Cardiovascular Disease Prediction API
All functionality in one file - no external dependencies
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger():
    """Setup application logger"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('cardio_app')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    
    # File handler
    log_file = os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()

# ============================================================================
# MODEL MANAGER CLASS
# ============================================================================

class ModelManager:
    """Manages ML model loading, training, and predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_type = None
        self.model_path = None
        self.model_info = {}
        self.load_default_model()
    
    def load_default_model(self):
        """Load the default or latest trained model"""
        try:
            trained_dir = os.path.join('models', 'trained')
            if os.path.exists(trained_dir):
                pkl_files = [f for f in os.listdir(trained_dir) if f.endswith('.pkl')]
                if pkl_files:
                    latest_model = max(pkl_files, key=lambda f: os.path.getmtime(os.path.join(trained_dir, f)))
                    model_path = os.path.join(trained_dir, latest_model)
                    self.load_model(model_path)
                    logger.info(f"Loaded model: {latest_model}")
                else:
                    logger.warning("No trained models found - will need to train first")
            else:
                logger.warning("Models directory not found - will need to train first")
        except Exception as e:
            logger.error(f"Error loading default model: {str(e)}")
    
    def load_model(self, model_path):
        """Load a trained model from disk"""
        try:
            model_data = joblib.load(model_path)
            
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.feature_names = model_data.get('feature_names')
                self.model_type = model_data.get('model_type')
            else:
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
            
            logger.info(f"Model loaded: {type(self.model).__name__}")
            if self.feature_names:
                logger.info(f"Features: {len(self.feature_names)} features")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def is_model_loaded(self):
        """Check if model is loaded and ready"""
        return self.model is not None
    
    def get_model_info(self):
        """Get information about current model"""
        return {
            'loaded': self.is_model_loaded(),
            'type': type(self.model).__name__ if self.model else None,
            'features': len(self.feature_names) if self.feature_names else None,
            'has_scaler': self.scaler is not None,
            'loaded_at': self.model_info.get('loaded_at')
        }
    
    def predict(self, input_data):
        """Make predictions using the loaded model"""
        if self.model is None:
            raise ValueError("No model loaded. Please train a model first.")
        
        try:
            # Convert input to DataFrame
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = pd.DataFrame(input_data)
            
            # Feature engineering
            if 'age' in df.columns and 'age_years' not in df.columns:
                df['age_years'] = df['age'] / 365.25
                df = df.drop('age', axis=1)
            
            if 'bmi' not in df.columns and 'height' in df.columns and 'weight' in df.columns:
                df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
            
            if 'pulse_pressure' not in df.columns and 'ap_hi' in df.columns and 'ap_lo' in df.columns:
                df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
            
            # Ensure features are in correct order
            if self.feature_names:
                missing_features = set(self.feature_names) - set(df.columns)
                if missing_features:
                    raise ValueError(f"Missing features: {missing_features}")
                df = df[self.feature_names]
            
            # Apply scaling if available
            if self.scaler is not None:
                df_scaled = self.scaler.transform(df)
                df = pd.DataFrame(df_scaled, columns=df.columns)
            
            # Make prediction
            prediction = self.model.predict(df)
            
            result = {
                'prediction': int(prediction[0]),
                'risk_level': 'High Risk' if prediction[0] == 1 else 'Low Risk'
            }
            
            # Get probability if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(df)
                result['probability'] = float(probabilities[0][1])
                result['confidence'] = float(max(probabilities[0]))
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def train_model(self, model_type='logistic_regression', data_path=None):
        """Train a new model"""
        try:
            logger.info(f"Starting training for {model_type} model")
            
            # Load data
            if data_path is None:
                data_path = os.path.join('data', 'cardio_train.csv')
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            # Read CSV
            df = pd.read_csv(data_path, sep=';')
            logger.info(f"Data loaded: {len(df)} samples")
            
            # Drop id column
            if 'id' in df.columns:
                df = df.drop('id', axis=1)
            
            # Feature engineering
            if 'age' in df.columns:
                df['age_years'] = df['age'] / 365.25
                df = df.drop('age', axis=1)
            
            if 'height' in df.columns and 'weight' in df.columns:
                df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
            
            if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
                df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
            
            # Remove outliers
            df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 220)]
            df = df[(df['ap_lo'] >= 50) & (df['ap_lo'] <= 140)]
            df = df[(df['height'] >= 140) & (df['height'] <= 210)]
            df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]
            
            logger.info(f"After preprocessing: {len(df)} samples")
            
            # Split features and target
            X = df.drop('cardio', axis=1)
            y = df['cardio']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            
            # Get model
            models = {
                'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'decision_tree': DecisionTreeClassifier(random_state=42),
                'svm': SVC(kernel='rbf', probability=True, random_state=42)
            }
            
            model = models.get(model_type, models['logistic_regression'])
            logger.info(f"Training {model.__class__.__name__}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            test_predictions = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_predictions)
            precision = precision_score(y_test, test_predictions)
            recall = recall_score(y_test, test_predictions)
            f1 = f1_score(y_test, test_predictions)
            
            logger.info(f"Accuracy: {test_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"cardio_model_{model_type}_{timestamp}.pkl"
            model_path = os.path.join('models', 'trained', model_filename)
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            model_data = {
                'model': model,
                'scaler': scaler,
                'feature_names': list(X_train.columns),
                'model_type': model_type
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved: {model_path}")
            
            # Load the new model
            self.load_model(model_path)
            
            return {
                'success': True,
                'model_path': model_path,
                'metrics': {
                    'test_accuracy': float(test_accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1)
                }
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

# ============================================================================
# RECOMMENDATION GENERATOR
# ============================================================================

def generate_recommendations(risk_level, risk_probability, patient_data):
    """Generate personalized recommendations based on risk factors"""
    recommendations = []
    risk_percentage = risk_probability * 100
    risk_factors = []
    
    # Critical risk (>90%)
    if risk_percentage >= 90:
        priority = 'critical'
        recommendations.append({
            'priority': priority,
            'category': 'Immediate Action Required',
            'title': '🚨 URGENT: Critical CVD Risk Detected',
            'description': f'Your risk score of {risk_percentage:.1f}% indicates CRITICAL cardiovascular disease risk. This requires IMMEDIATE medical attention. Visit the emergency room or call emergency services if experiencing chest pain, shortness of breath, or extreme fatigue. Schedule urgent cardiology consultation within 24-48 hours.'
        })
    
    # Very high risk (80-90%)
    elif risk_percentage >= 80:
        priority = 'critical'
        recommendations.append({
            'priority': priority,
            'category': 'Urgent Medical Attention',
            'title': '⚠️ Very High CVD Risk - Immediate Action Needed',
            'description': f'Your {risk_percentage:.1f}% risk level requires immediate medical intervention. Schedule an urgent appointment with a cardiologist within 1 week. Medication, lifestyle changes, and close monitoring are essential to prevent cardiovascular events.'
        })
    
    # High risk (66-80%)
    elif risk_percentage >= 66:
        priority = 'critical'
        recommendations.append({
            'priority': priority,
            'category': 'High Risk Alert',
            'title': '🔴 High CVD Risk Requires Action',
            'description': f'At {risk_percentage:.1f}% risk, you have a significantly elevated chance of cardiovascular disease. See a doctor within 2 weeks for comprehensive cardiac evaluation. Aggressive lifestyle modifications and likely medication are needed.'
        })
    
    # Elevated risk (50-66%)
    elif risk_percentage >= 50:
        priority = 'high'
        recommendations.append({
            'priority': priority,
            'category': 'Elevated Risk',
            'title': '🟠 Elevated CVD Risk - Take Action',
            'description': f'Your {risk_percentage:.1f}% risk indicates elevated cardiovascular disease risk. Schedule a doctor\'s appointment within 1 month for full evaluation. Start implementing lifestyle changes immediately: diet, exercise, stress management.'
        })
    
    # Moderate-high risk (40-50%)
    elif risk_percentage >= 40:
        priority = 'high'
        recommendations.append({
            'priority': priority,
            'category': 'Moderate-High Risk',
            'title': '🟡 Moderate-High Risk Level',
            'description': f'At {risk_percentage:.1f}% risk, you should take preventive action. Schedule a check-up within 2-3 months. Focus on healthy lifestyle: regular exercise (150 min/week), balanced diet, maintain healthy weight, manage stress.'
        })
    
    # Moderate risk (30-40%)
    elif risk_percentage >= 30:
        priority = 'medium'
        recommendations.append({
            'priority': priority,
            'category': 'Moderate Risk',
            'title': 'Moderate Risk - Prevention Focus',
            'description': f'Your {risk_percentage:.1f}% risk suggests room for improvement. Focus on prevention: eat more vegetables and fruits, exercise regularly, maintain healthy weight, avoid smoking, limit alcohol. Annual check-ups recommended.'
        })
    
    # Low-moderate risk (20-30%)
    elif risk_percentage >= 20:
        priority = 'medium'
        recommendations.append({
            'priority': priority,
            'category': 'Low-Moderate Risk',
            'title': 'Low-Moderate Risk - Stay Healthy',
            'description': f'At {risk_percentage:.1f}% risk, you\'re in a relatively good position. Maintain healthy habits: balanced diet, regular physical activity (30 min/day), stress management, adequate sleep. Regular health screenings recommended.'
        })
    
    # Low risk (<20%)
    else:
        priority = 'low'
        recommendations.append({
            'priority': priority,
            'category': 'Low Risk',
            'title': '✅ Low CVD Risk - Maintain Health',
            'description': f'Excellent! Your {risk_percentage:.1f}% risk indicates low cardiovascular disease risk. Continue your healthy lifestyle: stay active, eat nutritious foods, maintain healthy weight, avoid smoking, limit alcohol. Keep up the good work!'
        })
    
    # Physical activity recommendations
    if patient_data.get('active', 0) == 0:
        risk_factors.append('sedentary lifestyle')
        priority = 'critical' if risk_percentage >= 66 else 'high' if risk_percentage >= 40 else 'medium'
        recommendations.append({
            'priority': priority,
            'category': 'Exercise Program',
            'title': '💪 Start Physical Activity Immediately',
            'description': f'Sedentary lifestyle with {risk_percentage:.1f}% risk requires urgent action. Start with 10-minute walks 3x daily, gradually increasing to 30 minutes of moderate activity 5 days/week. Exercise can reduce CVD risk by 30-40%.'
        })
    
    # BMI-based recommendations
    height_m = patient_data.get('height', 170) / 100
    weight = patient_data.get('weight', 70)
    bmi = weight / (height_m ** 2)
    
    if bmi > 30:
        risk_factors.append('obesity')
        priority = 'critical' if risk_percentage >= 66 else 'high'
        recommendations.append({
            'priority': priority,
            'category': 'Weight Management',
            'title': '⚖️ Urgent Weight Loss Required',
            'description': f'BMI {bmi:.1f} (obese) + {risk_percentage:.1f}% CVD risk is dangerous. Target: lose 10% body weight ({weight*0.1:.1f}kg) in 6 months. Work with dietitian for 500-750 calorie/day deficit. Even 5-10% weight loss reduces risk significantly.'
        })
    elif bmi > 25:
        priority = 'medium'
        recommendations.append({
            'priority': priority,
            'category': 'Weight Management',
            'title': 'Weight Reduction Recommended',
            'description': f'BMI {bmi:.1f} (overweight) contributes to your {risk_percentage:.1f}% risk. Target BMI: 25 or below. Aim for 0.5-1kg weight loss per week through balanced diet and regular exercise.'
        })
    
    # Blood pressure recommendations
    ap_hi = patient_data.get('ap_hi', 120)
    ap_lo = patient_data.get('ap_lo', 80)
    
    if ap_hi >= 160 or ap_lo >= 100:
        risk_factors.append('hypertension stage 2')
        recommendations.append({
            'priority': 'critical',
            'category': 'Blood Pressure Emergency',
            'title': '🔴 Critical Hypertension - Immediate Action',
            'description': f'BP {ap_hi}/{ap_lo} + {risk_percentage:.1f}% risk = HYPERTENSIVE CRISIS. See doctor TODAY or visit ER if >180/120. Start medication immediately. Monitor BP twice daily. Reduce sodium to <1500mg/day.'
        })
    elif ap_hi >= 140 or ap_lo >= 90:
        risk_factors.append('hypertension stage 1')
        priority = 'critical' if risk_percentage >= 66 else 'high'
        recommendations.append({
            'priority': priority,
            'category': 'Blood Pressure Management',
            'title': 'Hypertension Treatment Needed',
            'description': f'BP {ap_hi}/{ap_lo} requires medication at {risk_percentage:.1f}% risk. See doctor within 1 week. Start DASH diet (reduce sodium <2300mg/day, increase potassium). Monitor BP daily.'
        })
    elif ap_hi >= 130 or ap_lo >= 85:
        priority = 'high' if risk_percentage >= 50 else 'medium'
        recommendations.append({
            'priority': priority,
            'category': 'Blood Pressure',
            'title': 'Elevated Blood Pressure Management',
            'description': f'BP {ap_hi}/{ap_lo} is elevated. At {risk_percentage:.1f}% risk, lifestyle changes needed: reduce sodium to <2000mg/day, lose weight if overweight, exercise 30 min/day, limit alcohol, manage stress.'
        })
    
    # Cholesterol recommendations
    if patient_data.get('cholesterol', 1) >= 2:
        risk_factors.append('high cholesterol')
        priority = 'critical' if risk_percentage >= 66 else 'high'
        recommendations.append({
            'priority': priority,
            'category': 'Cholesterol Management',
            'title': 'High Cholesterol Requires Treatment',
            'description': f'Elevated cholesterol with {risk_percentage:.1f}% risk requires intervention. See doctor for lipid panel. Reduce saturated fat, increase fiber, consider statin medication. Target LDL <100 mg/dL.'
        })
    
    # Smoking recommendations
    if patient_data.get('smoke', 0) == 1:
        risk_factors.append('smoking')
        recommendations.append({
            'priority': 'critical',
            'category': 'Smoking Cessation',
            'title': '🚭 QUIT SMOKING IMMEDIATELY',
            'description': f'Smoking + {risk_percentage:.1f}% CVD risk is life-threatening. Quitting is THE MOST IMPORTANT action. Talk to doctor about nicotine replacement, medications (Chantix, Wellbutrin), counseling. Quitting reduces heart disease risk by 50% within 1 year.'
        })
    
    # Alcohol recommendations
    if patient_data.get('alco', 0) == 1:
        priority = 'high' if risk_percentage >= 50 else 'medium'
        recommendations.append({
            'priority': priority,
            'category': 'Alcohol Management',
            'title': 'Reduce Alcohol Consumption',
            'description': f'Alcohol intake increases your {risk_percentage:.1f}% risk. Limit to ≤1 drink/day for women, ≤2 for men. Consider abstaining if risk is high. Excessive alcohol raises blood pressure and triglycerides.'
        })
    
    # Glucose/diabetes recommendations
    if patient_data.get('gluc', 1) >= 2:
        risk_factors.append('high glucose')
        priority = 'critical' if risk_percentage >= 66 else 'high'
        recommendations.append({
            'priority': priority,
            'category': 'Blood Sugar Management',
            'title': 'Glucose Control Required',
            'description': f'Elevated glucose with {risk_percentage:.1f}% CVD risk requires immediate action. Get HbA1c test. If diabetic, strict glucose control essential. Reduce refined carbs, increase fiber, monitor blood sugar. Diabetes doubles CVD risk.'
        })
    
    # General recommendations
    recommendations.append({
        'priority': 'medium',
        'category': 'Diet',
        'title': '🥗 Heart-Healthy Diet',
        'description': 'Follow Mediterranean or DASH diet: more vegetables, fruits, whole grains, fish, nuts, olive oil. Less red meat, processed foods, added sugars, saturated fat. Aim for 5+ servings of fruits/vegetables daily.'
    })
    
    recommendations.append({
        'priority': 'medium',
        'category': 'Monitoring',
        'title': '📊 Regular Health Monitoring',
        'description': f'With {risk_percentage:.1f}% risk, monitor key metrics: BP monthly, weight weekly, cholesterol annually, glucose annually. Keep health log. Schedule regular check-ups with your doctor.'
    })
    
    # Sort by priority
    priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
    
    return recommendations

# ============================================================================
# FLASK APPLICATION
# ============================================================================

app = Flask(__name__)
CORS(app)

# Initialize model manager
model_manager = ModelManager()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_manager.is_model_loaded(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make CVD risk prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        if not model_manager.is_model_loaded():
            return jsonify({'success': False, 'error': 'Model not loaded. Please train a model first.'}), 503
        
        logger.info(f"Prediction request: {data}")
        
        # Transform age from years to days for processing
        transformed_data = data.copy()
        if 'age' in transformed_data and transformed_data['age'] < 200:
            transformed_data['age'] = transformed_data['age'] * 365.25
        
        logger.info(f"Transformed data: {transformed_data}")
        
        # Make prediction
        prediction = model_manager.predict(transformed_data)
        
        logger.info(f"Prediction: {prediction}")
        
        # Generate recommendations
        recommendations = generate_recommendations(
            prediction.get('risk_level', 'Unknown'),
            prediction.get('probability', 0),
            data
        )
        
        model_info = model_manager.get_model_info()
        
        return jsonify({
            'success': True,
            'prediction': prediction.get('prediction', 0),
            'risk_level': prediction.get('risk_level', 'Unknown'),
            'risk_probability': prediction.get('probability', 0),
            'probability': prediction.get('probability', 0),
            'confidence': prediction.get('confidence', 0),
            'model_used': model_info.get('type', 'unknown'),
            'input': data,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get dataset statistics"""
    try:
        data_path = os.path.join('data', 'cardio_train.csv')
        
        if not os.path.exists(data_path):
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404
        
        # Load data
        df = pd.read_csv(data_path, delimiter=';')
        
        # Calculate statistics
        total_records = int(len(df))
        positive_cases = int(df[df['cardio'] == 1].shape[0])
        negative_cases = int(df[df['cardio'] == 0].shape[0])
        prevalence_value = float(positive_cases) / float(total_records) if total_records > 0 else 0.0
        
        # Age distribution
        df['age_years'] = df['age'] / 365.25
        age_bins = [0, 40, 50, 60, 70, 120]
        age_labels = ['30-40', '40-50', '50-60', '60-70', '70+']
        df['age_group'] = pd.cut(df['age_years'], bins=age_bins, labels=age_labels, right=False)
        age_dist = df['age_group'].value_counts().sort_index()
        age_distribution = {str(k): int(v) for k, v in age_dist.items()}
        
        # Statistics
        avg_age = float(df['age'].mean() / 365.25)
        female_count = int(df[df['gender'] == 1].shape[0])
        male_count = int(df[df['gender'] == 2].shape[0])
        avg_systolic = float(df['ap_hi'].mean())
        avg_diastolic = float(df['ap_lo'].mean())
        bmi_series = df['weight'] / ((df['height'] / 100) ** 2)
        avg_bmi = float(bmi_series.mean())
        
        stats = {
            'success': True,
            'total_samples': total_records,
            'total_records': total_records,
            'positive_cases': positive_cases,
            'negative_cases': negative_cases,
            'prevalence': round(prevalence_value, 4),
            'average_age': round(avg_age, 2),
            'age_distribution': age_distribution,
            'gender_distribution': {
                'female': female_count,
                'male': male_count
            },
            'average_systolic_bp': round(avg_systolic, 2),
            'average_diastolic_bp': round(avg_diastolic, 2),
            'avg_bmi': round(avg_bmi, 2)
        }
        
        logger.info(f"Stats: Total={total_records}, CVD={positive_cases}, Prev={prevalence_value:.2%}, AvgAge={avg_age:.1f}")
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train():
    """Train a new model"""
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'logistic_regression') if data else 'logistic_regression'
        
        logger.info(f"Training request: {model_type}")
        
        result = model_manager.train_model(model_type)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': 'Model trained successfully',
                'metrics': result.get('metrics', {}),
                'model_path': result['model_path']
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Training failed')
            }), 500
            
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    try:
        trained_dir = os.path.join('models', 'trained')
        models = []
        
        if os.path.exists(trained_dir):
            pkl_files = [f for f in os.listdir(trained_dir) if f.endswith('.pkl')]
            models = sorted(pkl_files, key=lambda f: os.path.getmtime(os.path.join(trained_dir, f)), reverse=True)
        
        return jsonify({
            'success': True,
            'models': models,
            'current_model': os.path.basename(model_manager.model_path) if model_manager.model_path else None
        })
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("\n" + "="*80)
    print("CARDIOVASCULAR DISEASE PREDICTION API - STANDALONE VERSION")
    print("="*80)
    print(f"Server: http://localhost:{port}")
    print(f"Health: http://localhost:{port}/api/health")
    print(f"Predict: http://localhost:{port}/api/predict")
    print(f"Stats: http://localhost:{port}/api/stats")
    print("="*80)
    
    if model_manager.is_model_loaded():
        print("MODEL STATUS: READY")
        print(f"Model Type: {model_manager.get_model_info()['type']}")
        print(f"Features: {model_manager.get_model_info()['features']} features")
        print("FRONTEND CAN NOW CONNECT AND MAKE PREDICTIONS")
    else:
        print("MODEL STATUS: NOT LOADED")
        print("Train a model by calling: POST /api/train")
        print("Or ensure a trained model exists in models/trained/")
    
    print("="*80)
    print("Server is starting...")
    print("="*80 + "\n")
    
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
