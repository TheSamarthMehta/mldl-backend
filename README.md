# Cardiovascular Disease Prediction API

Flask backend for predicting cardiovascular disease risk using Logistic Regression.

## 🚀 Quick Setup

1. **Install Dependencies:**
   ```bash
   install.bat
   ```

2. **Train Model:**
   ```bash
   train.bat
   ```
   Or: `python scripts/train.py`

3. **Start Server:**
   ```bash
   run.bat
   ```
   Server runs on: `http://localhost:5000`

## 📡 API Endpoint for Frontend

**POST** `http://localhost:5000/api/predict`

**Request Body:**
```json
{
  "age": 50,
  "gender": "Male",
  "height": 168,
  "weight": 62,
  "ap_hi": 110,
  "ap_lo": 80,
  "cholesterol": "Normal",
  "glucose": "Normal",
  "smoke": false,
  "alco": false,
  "active": true
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "prediction": 0,
    "risk_level": "Low Risk",
    "probability": 0.15,
    "confidence": 0.85
  },
  "timestamp": "2026-01-13T17:00:00"
}
```

## 📊 Model Performance

- **Algorithm:** Logistic Regression
- **Training Data:** 68,552 patients (after outlier removal)
- **Test Accuracy:** 73.16%
- **Precision:** 75.86%
- **Recall:** 66.87%
- **F1-Score:** 71.08%
- **ROC AUC:** 79.58%

## 🔌 Frontend Integration

Your React frontend should:

1. **Set API URL:**
   ```javascript
   const API_URL = 'http://localhost:5000';
   ```

2. **Make POST request:**
   ```javascript
   const response = await fetch(`${API_URL}/api/predict`, {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify(formData)
   });
   
   const result = await response.json();
   
   if (result.success) {
     const { risk_level, probability, confidence } = result.prediction;
     // Display results to user
   }
   ```

## ✅ Ready for Production

- ✅ CORS enabled for frontend
- ✅ Input validation
- ✅ Error handling
- ✅ Logging system
- ✅ Model auto-loads latest trained version

## 📁 Project Structure

```
backend/
├── app.py                    # Flask API
├── requirements.txt          # Dependencies
├── install.bat              # Setup
├── run.bat                  # Start server
├── train.bat                # Train model
├── data/
│   └── cardio_train.csv     # 70,000 patient records
├── models/
│   ├── model_manager.py     # Model handler
│   └── trained/             # Trained models
├── scripts/
│   └── train.py            # Training script
└── utils/
    ├── logger.py           # Logging
    ├── validators.py       # Validation
    └── data_utils.py       # Data processing
```

## 🎯 Model Details

**Input Features (11):**
- age, gender, height, weight
- ap_hi, ap_lo (blood pressure)
- cholesterol, glucose
- smoke, alco, active

**Engineered Features (3):**
- age_years, bmi, pulse_pressure

**Output:**
- 0 = No cardiovascular disease (Low Risk)
- 1 = Cardiovascular disease present (High Risk)

---

**Everything is ready for frontend-backend connectivity!** 🎉
