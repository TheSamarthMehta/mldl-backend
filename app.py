from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger
from utils.validators import validate_prediction_input
from models.model_manager import ModelManager
from scripts.train import train_model

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
logger = setup_logger()

# Initialize model manager
model_manager = ModelManager()

# Check if model is loaded
if model_manager.is_model_loaded():
    logger.info("=" * 70)
    logger.info("MODEL IS READY FOR PREDICTIONS!")
    logger.info("=" * 70)
    model_info = model_manager.get_model_info()
    logger.info(f"Model Type: {model_info.get('type', 'Unknown')}")
    logger.info(f"Features: {len(model_info.get('feature_names', []))} features")
    logger.info(f"Loaded at: {model_info.get('loaded_at', 'Unknown')}")
    logger.info("=" * 70)
else:
    logger.warning("WARNING: No model loaded! Please train a model first: python scripts/train.py")

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['JSON_SORT_KEYS'] = False


def generate_recommendations(risk_level, probability, patient_data):
    """Generate personalized recommendations based on percentage risk ranges"""
    recommendations = []
    risk_factors = []
    risk_percentage = probability * 100
    
    # Determine risk category based on percentage ranges (same as gauge colors)
    if risk_percentage < 33:
        risk_category = 'low'
        urgency_level = 'low'
    elif risk_percentage < 66:
        risk_category = 'medium'
        urgency_level = 'medium'
    else:
        risk_category = 'high'
        urgency_level = 'critical' if risk_percentage >= 80 else 'high'
    
    # PRIMARY RECOMMENDATION based on precise risk percentage
    if risk_percentage >= 90:
        recommendations.append({
            'priority': 'critical',
            'category': 'Urgent Medical Care',
            'title': '🚨 Emergency Cardiovascular Evaluation Required',
            'description': f'Your risk is {risk_percentage:.1f}% - CRITICAL LEVEL. Visit emergency care or call your doctor TODAY. You need immediate ECG, cardiac enzyme tests, stress test, and comprehensive cardiovascular assessment within 24-48 hours.'
        })
    elif risk_percentage >= 80:
        recommendations.append({
            'priority': 'critical',
            'category': 'Urgent Medical Care',
            'title': 'Immediate Medical Attention Needed',
            'description': f'At {risk_percentage:.1f}% risk, seek medical care within 2-3 days. Schedule appointments for ECG, lipid panel, HbA1c test, and comprehensive cardiovascular screening. Consider emergency room if experiencing chest pain or shortness of breath.'
        })
    elif risk_percentage >= 70:
        recommendations.append({
            'priority': 'high',
            'category': 'Medical Consultation',
            'title': 'Urgent Doctor Appointment Required',
            'description': f'Your {risk_percentage:.1f}% risk requires medical intervention within 1 week. Schedule comprehensive cardiac evaluation including stress test, cholesterol screening, and blood pressure monitoring. Discuss preventive medications with your doctor.'
        })
    elif risk_percentage >= 60:
        recommendations.append({
            'priority': 'high',
            'category': 'Medical Consultation',
            'title': 'Schedule Medical Evaluation Soon',
            'description': f'At {risk_percentage:.1f}% risk, see your doctor within 2 weeks. Get full cardiovascular risk assessment, blood work (lipid panel, glucose), and blood pressure monitoring. Discuss lifestyle modifications and potential medication.'
        })
    elif risk_percentage >= 50:
        recommendations.append({
            'priority': 'high',
            'category': 'Medical Check-up',
            'title': 'Medical Check-up Recommended',
            'description': f'Your {risk_percentage:.1f}% risk is above average. Schedule a doctor visit within 1 month for cardiovascular screening, blood pressure check, cholesterol testing, and personalized prevention plan.'
        })
    elif risk_percentage >= 40:
        recommendations.append({
            'priority': 'medium',
            'category': 'Preventive Care',
            'title': 'Preventive Medical Consultation',
            'description': f'At {risk_percentage:.1f}% risk, schedule routine check-up within 2 months. Focus on lifestyle modifications, regular BP monitoring, and annual cardiovascular screening to prevent risk escalation.'
        })
    elif risk_percentage >= 33:
        recommendations.append({
            'priority': 'medium',
            'category': 'Preventive Care',
            'title': 'Routine Monitoring Advised',
            'description': f'Your {risk_percentage:.1f}% risk is borderline. Maintain regular check-ups every 6 months, adopt heart-healthy habits, and monitor blood pressure monthly to keep risk from increasing.'
        })
    elif risk_percentage >= 20:
        recommendations.append({
            'priority': 'low',
            'category': 'Prevention',
            'title': 'Maintain Heart-Healthy Lifestyle',
            'description': f'Your {risk_percentage:.1f}% risk is low. Continue your healthy habits with annual check-ups, regular exercise (150 min/week), balanced diet, and stress management to keep your heart healthy.'
        })
    else:
        recommendations.append({
            'priority': 'low',
            'category': 'Prevention',
            'title': 'Excellent Heart Health Status',
            'description': f'Your {risk_percentage:.1f}% risk is very low. Keep up your excellent lifestyle choices! Continue with annual wellness exams, stay physically active, and maintain your healthy diet and habits.'
        })
    
    # LIFESTYLE RISK FACTORS - with percentage-adjusted urgency
    if patient_data.get('smoke', 0) == 1:
        risk_factors.append('smoking')
        priority = 'critical' if risk_percentage >= 66 else 'high'
        recommendations.append({
            'priority': priority,
            'category': 'Critical Lifestyle Change',
            'title': '🚭 Quit Smoking IMMEDIATELY',
            'description': f'Smoking with {risk_percentage:.1f}% CVD risk is life-threatening. Quitting can reduce your risk by 50% within 1 year. Contact smoking cessation programs, use nicotine replacement therapy (patches, gum), or ask doctor about prescription medications like Chantix or Zyban.'
        })
    
    if patient_data.get('alco', 0) == 1:
        risk_factors.append('alcohol consumption')
        priority = 'high' if risk_percentage >= 66 else 'medium'
        recommendations.append({
            'priority': priority,
            'category': 'Lifestyle Modification',
            'title': '🍷 Reduce Alcohol Consumption',
            'description': f'At {risk_percentage:.1f}% risk, limit alcohol to maximum 1 drink/day (women) or 2 drinks/day (men). Consider complete abstinence if risk is high. Excessive alcohol raises blood pressure and triglycerides significantly.'
        })
    
    if patient_data.get('active', 1) == 0:
        risk_factors.append('physical inactivity')
        priority = 'high' if risk_percentage >= 50 else 'medium'
        recommendations.append({
            'priority': priority,
            'category': 'Exercise Program',
            'title': '💪 Start Physical Activity Immediately',
            'description': f'Sedentary lifestyle with {risk_percentage:.1f}% risk requires urgent action. Start with 10-minute walks 3x daily, gradually increasing to 30 minutes of moderate activity 5 days/week. Join cardiac rehab program if available. Exercise can reduce CVD risk by 30-40%.'
        })
    
    # BMI-BASED RECOMMENDATIONS with percentage context
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
            'description': f'BMI {bmi:.1f} (obese) + {risk_percentage:.1f}% CVD risk is dangerous. Target: lose 10% body weight ({weight*0.1:.1f}kg) in 6 months. Work with dietitian for 500-750 calorie/day deficit. Even 5-10% weight loss reduces risk significantly. Consider medical weight loss programs.'
        })
    elif bmi > 27:
        priority = 'high' if risk_percentage >= 50 else 'medium'
        recommendations.append({
            'priority': priority,
            'category': 'Weight Management',
            'title': 'Weight Reduction Recommended',
            'description': f'BMI {bmi:.1f} (overweight) contributes to your {risk_percentage:.1f}% risk. Target BMI: 25 or below. Aim for 0.5-1kg weight loss per week through balanced diet (focus on vegetables, lean protein, whole grains) and regular exercise.'
        })
    elif bmi > 25:
        recommendations.append({
            'priority': 'medium',
            'category': 'Weight Management',
            'title': 'Weight Monitoring Advised',
            'description': f'BMI {bmi:.1f} is slightly elevated. Maintain current weight or aim for modest reduction to BMI 22-24. Focus on portion control and regular physical activity to prevent weight gain.'
        })
    
    # BLOOD PRESSURE with percentage-based urgency
    ap_hi = patient_data.get('ap_hi', 120)
    ap_lo = patient_data.get('ap_lo', 80)
    
    if ap_hi >= 160 or ap_lo >= 100:
        risk_factors.append('hypertension stage 2')
        recommendations.append({
            'priority': 'critical',
            'category': 'Blood Pressure Emergency',
            'title': '🔴 Critical Hypertension - Immediate Action',
            'description': f'BP {ap_hi}/{ap_lo} + {risk_percentage:.1f}% risk = HYPERTENSIVE CRISIS. See doctor TODAY or visit ER if >180/120. Start medication immediately. Monitor BP twice daily. Reduce sodium to <1500mg/day. This requires urgent medical intervention to prevent heart attack or stroke.'
        })
    elif ap_hi >= 140 or ap_lo >= 90:
        risk_factors.append('hypertension stage 1')
        priority = 'critical' if risk_percentage >= 66 else 'high'
        recommendations.append({
            'priority': priority,
            'category': 'Blood Pressure Management',
            'title': 'Hypertension Treatment Needed',
            'description': f'BP {ap_hi}/{ap_lo} requires medication at {risk_percentage:.1f}% risk. See doctor within 1 week. Start DASH diet (reduce sodium <2300mg/day, increase potassium). Monitor BP daily. Likely need antihypertensive medication (ACE inhibitor, ARB, or CCB).'
        })
    elif ap_hi >= 130 or ap_lo >= 85:
        priority = 'high' if risk_percentage >= 50 else 'medium'
        recommendations.append({
            'priority': priority,
            'category': 'Blood Pressure',
            'title': 'Elevated Blood Pressure Management',
            'description': f'BP {ap_hi}/{ap_lo} is elevated. At {risk_percentage:.1f}% risk, aggressive lifestyle changes needed: reduce sodium to <2000mg/day, lose weight if overweight, exercise 30 min/day, limit alcohol, manage stress. Monitor BP weekly. May need medication if not improving in 3-6 months.'
        })
    elif ap_hi >= 120:
        recommendations.append({
            'priority': 'medium',
            'category': 'Blood Pressure',
            'title': 'Blood Pressure Monitoring',
            'description': f'BP {ap_hi}/{ap_lo} is prehypertensive. Monitor monthly and maintain healthy lifestyle to prevent progression. Focus on weight management, regular exercise, and low-sodium diet (<2300mg/day).'
        })
    
    # CHOLESTEROL with context
    if patient_data.get('cholesterol', 1) == 3:
        risk_factors.append('very high cholesterol')
        priority = 'critical' if risk_percentage >= 66 else 'high'
        recommendations.append({
            'priority': priority,
            'category': 'Cholesterol Management',
            'title': 'Severe Hypercholesterolemia Treatment',
            'description': f'Very high cholesterol + {risk_percentage:.1f}% risk requires immediate statin therapy. See doctor within 3 days. Get lipid panel (LDL, HDL, triglycerides). Target LDL <70mg/dL. Adopt Mediterranean diet, eliminate trans fats, increase fiber (oats, beans), take omega-3 supplements.'
        })
    elif patient_data.get('cholesterol', 1) == 2:
        risk_factors.append('elevated cholesterol')
        priority = 'high' if risk_percentage >= 50 else 'medium'
        recommendations.append({
            'priority': priority,
            'category': 'Cholesterol',
            'title': 'Cholesterol Reduction Program',
            'description': f'Elevated cholesterol at {risk_percentage:.1f}% risk needs aggressive management. Get lipid panel, likely need statin. Diet changes: limit saturated fat <7% calories, increase soluble fiber (10-25g/day), add plant sterols (2g/day). Recheck in 6 weeks.'
        })
    
    # GLUCOSE/DIABETES
    if patient_data.get('gluc', 1) == 3:
        risk_factors.append('very high blood sugar')
        priority = 'critical' if risk_percentage >= 66 else 'high'
        recommendations.append({
            'priority': priority,
            'category': 'Diabetes Management',
            'title': 'Urgent Diabetes Evaluation',
            'description': f'Very high glucose + {risk_percentage:.1f}% CVD risk is critical. Get HbA1c and fasting glucose tests immediately. Likely need diabetes medication (Metformin, etc.). See endocrinologist. Strict carb control (<130g/day), monitor blood sugar 3x/day, exercise after meals.'
        })
    elif patient_data.get('gluc', 1) == 2:
        risk_factors.append('elevated blood sugar')
        priority = 'high' if risk_percentage >= 50 else 'medium'
        recommendations.append({
            'priority': priority,
            'category': 'Blood Sugar',
            'title': 'Prediabetes Management',
            'description': f'Elevated glucose at {risk_percentage:.1f}% risk requires intervention. Get HbA1c test. Reduce refined carbs, sugar intake. Follow low-glycemic diet. Exercise 30 min after meals. Lose 7% body weight to reduce diabetes risk by 58%. Monitor fasting glucose monthly.'
        })
    
    # AGE-SPECIFIC with percentage awareness
    age = patient_data.get('age', 50)
    if age >= 65 and risk_percentage >= 50:
        recommendations.append({
            'priority': 'high',
            'category': 'Age-Related Monitoring',
            'title': 'Enhanced Monitoring for Age 65+',
            'description': f'Age {age} with {risk_percentage:.1f}% risk requires vigilant monitoring. Quarterly medical check-ups, monthly BP checks, annual cardiac stress test, echocardiogram every 2 years. Discuss aspirin therapy and aggressive risk factor management with doctor.'
        })
    elif age >= 60:
        priority = 'medium' if risk_percentage < 50 else 'high'
        recommendations.append({
            'priority': priority,
            'category': 'Preventive Care',
            'title': 'Increased Screening for Age 60+',
            'description': f'At age {age}, biannual cardiovascular check-ups recommended. Annual ECG, lipid panel, diabetes screening. Discuss preventive medications (statins, aspirin) with your doctor even if risk is currently {risk_percentage:.1f}%.'
        })
    
    # COMPREHENSIVE LIFESTYLE RECOMMENDATIONS based on risk tier
    if risk_percentage >= 66:
        recommendations.append({
            'priority': 'high',
            'category': 'Comprehensive Lifestyle',
            'title': '🎯 Intensive Risk Reduction Program',
            'description': 'HIGH RISK tier requires: 1) Immediate doctor visit, 2) Daily BP monitoring, 3) 40min exercise 6 days/week, 4) Strict heart-healthy diet (Mediterranean/DASH), 5) Stress management (meditation 15min/day), 6) Sleep 7-8 hours, 7) Weekly weight tracking, 8) Medication compliance.'
        })
    elif risk_percentage >= 33:
        recommendations.append({
            'priority': 'medium',
            'category': 'Lifestyle Optimization',
            'title': '📊 Moderate Risk Prevention Plan',
            'description': 'MEDIUM RISK tier: 1) Doctor check-up within month, 2) Weekly BP checks, 3) 150min moderate exercise/week, 4) Heart-healthy diet (more vegetables, less processed foods), 5) Weight management, 6) Stress reduction, 7) Limit alcohol, 8) Quality sleep 7-9 hours.'
        })
    else:
        recommendations.append({
            'priority': 'low',
            'category': 'Health Maintenance',
            'title': '✅ Maintain Excellent Heart Health',
            'description': 'LOW RISK tier: 1) Annual wellness check-up, 2) Continue regular exercise, 3) Maintain healthy diet, 4) Manage stress, 5) Healthy weight, 6) Moderate alcohol, 7) No smoking, 8) Good sleep habits. You\'re doing great - keep it up!'
        })
    
    return {
        'items': recommendations[:8],  # Top 8 most relevant recommendations
        'risk_factors_identified': risk_factors,
        'risk_percentage': risk_percentage,
        'risk_category': risk_category,
        'total_recommendations': len(recommendations)
    }


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'Backend API is running',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_manager.is_model_loaded(),
        'models_loaded': model_manager.is_model_loaded(),
        'available_models': model_manager.get_available_models()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions using the loaded model"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        logger.info(f"Received prediction request: {data}")
        
        # Transform frontend data to match model expectations
        # Frontend can send either numeric (1, 2, 3) or text ("Male", "Normal") formats
        
        # Handle gender - can be numeric (1, 2) or text ("Male", "Female")
        gender = data.get('gender', 1)
        if isinstance(gender, str):
            gender_map = {'Male': 2, 'Female': 1, 'male': 2, 'female': 1}
            gender = gender_map.get(gender, 1)
        else:
            gender = int(gender)
        
        # Handle cholesterol - can be numeric (1, 2, 3) or text
        cholesterol = data.get('cholesterol', 1)
        if isinstance(cholesterol, str):
            level_map = {
                'Normal': 1,
                'Above Normal': 2,
                'Well Above Normal': 3,
                'normal': 1,
                'above normal': 2,
                'well above normal': 3
            }
            cholesterol = level_map.get(cholesterol, 1)
        else:
            cholesterol = int(cholesterol)
        
        # Handle glucose - can be numeric (1, 2, 3) or text
        glucose = data.get('gluc', data.get('glucose', 1))
        if isinstance(glucose, str):
            level_map = {
                'Normal': 1,
                'Above Normal': 2,
                'Well Above Normal': 3,
                'normal': 1,
                'above normal': 2,
                'well above normal': 3
            }
            glucose = level_map.get(glucose, 1)
        else:
            glucose = int(glucose)
        
        # Extract and transform data
        transformed_data = {
            'age': float(data.get('age', 50)) * 365.25,  # Convert years to days for model
            'gender': gender,
            'height': float(data.get('height', 170)),
            'weight': float(data.get('weight', 70)),
            'ap_hi': float(data.get('ap_hi', 120)),
            'ap_lo': float(data.get('ap_lo', 80)),
            'cholesterol': cholesterol,
            'gluc': glucose,
            'smoke': int(data.get('smoke', 0)),
            'alco': int(data.get('alco', 0)),
            'active': int(data.get('active', 1))
        }
        
        logger.info(f"Transformed data: {transformed_data}")
        
        # Validate input
        is_valid, error_message = validate_prediction_input(transformed_data)
        if not is_valid:
            logger.error(f"Validation failed: {error_message}")
            return jsonify({
                'success': False,
                'error': error_message
            }), 400
        
        # Make prediction
        prediction = model_manager.predict(transformed_data)
        
        logger.info(f"Prediction made successfully: {prediction}")
        
        # Get model name
        model_info = model_manager.get_model_info()
        model_name = model_info.get('type', 'logistic_regression')
        
        # Generate personalized recommendations based on risk and patient data
        recommendations = generate_recommendations(
            prediction.get('risk_level', 'Unknown'),
            prediction.get('probability', 0),
            data
        )
        
        return jsonify({
            'success': True,
            'prediction': prediction.get('prediction', 0),
            'risk_level': prediction.get('risk_level', 'Unknown'),
            'risk_probability': prediction.get('probability', 0),
            'probability': prediction.get('probability', 0),
            'confidence': prediction.get('confidence', 0),
            'model_used': model_name,
            'input': data,  # Return the original input data
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    try:
        models = model_manager.get_available_models()
        return jsonify({
            'success': True,
            'models': models,
            'current_model': model_manager.get_current_model()
        })
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get information about the current model"""
    try:
        info = model_manager.get_model_info()
        return jsonify({
            'success': True,
            'model_info': info
        })
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/data/info', methods=['GET'])
def get_data_info():
    """Get information about the dataset"""
    try:
        info = {
            'dataset': 'Cardiovascular Disease',
            'total_records': 70000,
            'features': {
                'age': 'Age in days (converted to years)',
                'gender': 'Gender (1: Female, 2: Male)',
                'height': 'Height in cm',
                'weight': 'Weight in kg',
                'ap_hi': 'Systolic blood pressure',
                'ap_lo': 'Diastolic blood pressure',
                'cholesterol': 'Cholesterol level (1: Normal, 2: Above normal, 3: Well above normal)',
                'gluc': 'Glucose level (1: Normal, 2: Above normal, 3: Well above normal)',
                'smoke': 'Smoking (0: No, 1: Yes)',
                'alco': 'Alcohol intake (0: No, 1: Yes)',
                'active': 'Physical activity (0: No, 1: Yes)'
            },
            'engineered_features': {
                'age_years': 'Age in years',
                'bmi': 'Body Mass Index',
                'pulse_pressure': 'Pulse pressure (ap_hi - ap_lo)'
            },
            'target': {
                'cardio': 'Cardiovascular disease (0: No, 1: Yes)'
            },
            'example': {
                'age': 18393,
                'gender': 2,
                'height': 168,
                'weight': 62.0,
                'ap_hi': 110,
                'ap_lo': 80,
                'cholesterol': 1,
                'gluc': 1,
                'smoke': 0,
                'alco': 0,
                'active': 1
            }
        }
        return jsonify({
            'success': True,
            'data_info': info
        })
    except Exception as e:
        logger.error(f"Error getting data info: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/train', methods=['POST'])
def train():
    """Train a new model"""
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'logistic_regression') if data else 'logistic_regression'
        
        logger.info(f"Starting training for model type: {model_type}")
        
        # Train model
        result = train_model(model_type)
        
        if result['success']:
            # Reload model manager
            model_manager.load_model(result['model_path'])
            
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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Evaluate model performance"""
    try:
        data = request.get_json()
        
        metrics = model_manager.evaluate(data)
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get dataset statistics for analytics page"""
    try:
        # Check if data file exists
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'cardio_train.csv')
        
        if not os.path.exists(data_path):
            return jsonify({
                'success': False,
                'error': 'Dataset not found'
            }), 404
        
        # Load and analyze data
        import pandas as pd
        import numpy as np
        df = pd.read_csv(data_path, delimiter=';')
        
        # Calculate counts
        total_records = int(len(df))
        positive_cases = int(df[df['cardio'] == 1].shape[0])
        negative_cases = int(df[df['cardio'] == 0].shape[0])
        prevalence_value = float(positive_cases) / float(total_records) if total_records > 0 else 0.0
        
        # Calculate age distribution
        df['age_years'] = df['age'] / 365.25
        age_bins = [0, 40, 50, 60, 70, 120]
        age_labels = ['30-40', '40-50', '50-60', '60-70', '70+']
        df['age_group'] = pd.cut(df['age_years'], bins=age_bins, labels=age_labels, right=False)
        age_dist = df['age_group'].value_counts().sort_index()
        age_distribution = {str(k): int(v) for k, v in age_dist.items()}
        
        # Age statistics (convert from days to years)
        avg_age = float(df['age'].mean() / 365.25)
        
        # Gender distribution (1=Female, 2=Male)
        female_count = int(df[df['gender'] == 1].shape[0])
        male_count = int(df[df['gender'] == 2].shape[0])
        
        # Blood pressure statistics
        avg_systolic = float(df['ap_hi'].mean())
        avg_diastolic = float(df['ap_lo'].mean())
        
        # BMI statistics
        bmi_series = df['weight'] / ((df['height'] / 100) ** 2)
        avg_bmi = float(bmi_series.mean())
        
        # Lifestyle factors
        smokers_count = int(df[df['smoke'] == 1].shape[0])
        alcohol_count = int(df[df['alco'] == 1].shape[0])
        active_count = int(df[df['active'] == 1].shape[0])
        
        # Cholesterol distribution
        chol_normal = int(df[df['cholesterol'] == 1].shape[0])
        chol_above = int(df[df['cholesterol'] == 2].shape[0])
        chol_high = int(df[df['cholesterol'] == 3].shape[0])
        
        # Glucose distribution
        gluc_normal = int(df[df['gluc'] == 1].shape[0])
        gluc_above = int(df[df['gluc'] == 2].shape[0])
        gluc_high = int(df[df['gluc'] == 3].shape[0])
        
        # Build response with explicit typing
        stats = {
            'success': True,
            'total_samples': total_records,
            'total_records': total_records,
            'positive_cases': positive_cases,
            'negative_cases': negative_cases,
            'prevalence': round(prevalence_value, 4),
            'features_count': len(df.columns) - 1,
            
            # Age statistics
            'average_age': round(avg_age, 2),
            'age_mean': round(avg_age, 2),
            'age_min': round(float(df['age'].min() / 365.25), 2),
            'age_max': round(float(df['age'].max() / 365.25), 2),
            'age_distribution': age_distribution,
            
            # Gender distribution
            'gender_distribution': {
                'female': female_count,
                'male': male_count
            },
            
            # Blood pressure statistics
            'avg_systolic_bp': round(avg_systolic, 2),
            'avg_diastolic_bp': round(avg_diastolic, 2),
            'average_systolic_bp': round(avg_systolic, 2),
            'average_diastolic_bp': round(avg_diastolic, 2),
            'ap_hi_mean': round(avg_systolic, 2),
            'ap_lo_mean': round(avg_diastolic, 2),
            
            # BMI statistics
            'avg_bmi': round(avg_bmi, 2),
            'bmi_mean': round(avg_bmi, 2),
            
            # Lifestyle factors
            'smokers': smokers_count,
            'alcohol': alcohol_count,
            'active': active_count,
            
            # Cholesterol distribution
            'cholesterol_distribution': {
                'normal': chol_normal,
                'above_normal': chol_above,
                'well_above_normal': chol_high
            },
            
            # Glucose distribution
            'glucose_distribution': {
                'normal': gluc_normal,
                'above_normal': gluc_above,
                'well_above_normal': gluc_high
            }
        }
        
        logger.info(f"Statistics calculated: Total={total_records}, CVD={positive_cases}, Prev={prevalence_value:.2%}, AvgAge={avg_age:.1f}")
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("\n" + "="*70)
    print("CARDIOVASCULAR DISEASE PREDICTION API")
    print("="*70)
    print(f"Server: http://localhost:{port}")
    print(f"API Endpoint: http://localhost:{port}/api/predict")
    print(f"Health Check: http://localhost:{port}/api/health")
    print("="*70)
    
    if model_manager.is_model_loaded():
        print("MODEL STATUS: READY")
        print("FRONTEND CAN NOW CONNECT AND MAKE PREDICTIONS")
    else:
        print("MODEL STATUS: NOT LOADED")
        print("Run: python scripts/train.py")
    
    print("="*70)
    print("Server is starting...")
    print("="*70 + "\n")
    
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
