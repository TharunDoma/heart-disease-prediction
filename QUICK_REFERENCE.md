# 🚀 Quick Reference Guide

## Setup & Run (30 seconds)

```bash
cd "c:\Users\tharu\Downloads\main project"
pip install -r requirements.txt
python app.py
```

Then open: **http://127.0.0.1:5000**

---

## Project at a Glance

| Aspect | Details |
|--------|---------|
| **Type** | Web-based ML Application |
| **ML Model** | Deep Neural Network (3 layers) |
| **Input** | 20 medical parameters |
| **Output** | Heart disease prediction + confidence % |
| **Language** | Python + HTML/CSS/JS |
| **Framework** | Flask |
| **Training Data** | 335 patient records (ECG dataset) |
| **Prediction Time** | <100ms |

---

## File Purposes

| File | Purpose |
|------|---------|
| **app.py** | Main Flask application with all routes |
| **model.py** | Train the neural network (run once) |
| **index.html** | Patient data input form |
| **result.html** | Results, graphs, and report download |
| **ECG-Dataset.csv** | Training data (335 samples) |
| **heart_disease_model.h5** | Trained neural network model |
| **scaler.save** | Feature standardizer (pickle) |
| **DEVELOPER_GUIDE.md** | Complete technical documentation |
| **README.md** | Project overview and features |
| **requirements.txt** | Python dependencies |

---

## How the Algorithm Works

```
Step 1: User enters 20 medical values
         ↓
Step 2: Values standardized using saved scaler
         (normalize to mean=0, std=1)
         ↓
Step 3: Feed through neural network
         Input(20) → Dense(64+ReLU) → Dense(32+ReLU) → Output(1+Sigmoid)
         ↓
Step 4: Get probability (0.0 to 1.0)
         ↓
Step 5: Convert to prediction
         If prob > 0.5 → POSITIVE (has disease)
         If prob ≤ 0.5 → NEGATIVE (no disease)
         ↓
Step 6: Calculate confidence
         Confidence = abs(prob - 0.5) × 2 × 100 %
```

---

## 20 Medical Input Features

### Demographics (2)
- Age (20-90 years)
- Sex (Female/Male)

### Lifestyle (3)
- Smoker (Yes/No)
- Years of smoking (0-50)
- Physical activity (Yes/No)

### Clinical (8)
- Chest pain type (1-4)
- ECG pattern (1-4)
- Q wave (Yes/No)
- Previous heart intervention (Yes/No)
- Diabetes (Yes/No)
- Hypertension (Yes/No)
- Family history (Yes/No)
- Interventricular septal defect (Yes/No)

### Vitals (5)
- Heart rate (40-140 bpm)
- Systolic BP (80-220 mmHg)
- Diastolic BP (40-140 mmHg)
- LDL cholesterol (26-260 mg/dL)
- Height (128-192 cm) & Weight (41-134 kg)

---

## API Endpoints

### GET /
```
Returns: index.html (input form page)
```

### POST /predict
```
Input: Form data with 20 feature values
Returns: result.html with prediction & graphs
```

### POST /download-report
```
Input: JSON with patient data
Returns: Word document (.docx) file
```

---

## Example Prediction

**Patient Input:**
```
Age: 55
Sex: Male
Smoker: No
LDL: 113
Heart Rate: 84
BP: 124/75
Diabetes: No
Hypertension: No
... (14 more features)
```

**Neural Network Processing:**
```
Raw features → Standardized → Layer1(64) → Layer2(32) → Output(1)
Result: 0.35 probability
```

**Final Output:**
```
✓ NEGATIVE: No heart disease
Confidence: 65%
Risk Level: LOW RISK
```

---

## Visualization Outputs

### Graph 1: Patient Metrics Dashboard
- Age, LDL, Heart Rate, Systolic BP, BMI (colored by risk level)
- Risk factors summary box

### Graph 2: Feature Correlations
- Heatmap showing relationships between key health metrics
- Correlation with disease prediction

### Graph 3: Risk Distribution
- Bar chart: Healthy vs diseased patients in training data
- Percentage breakdown

---

## Common Issues & Fixes

| Problem | Solution |
|---------|----------|
| ModuleNotFoundError | `pip install -r requirements.txt` |
| Port 5000 in use | Change port: edit app.py line 217 |
| No graphs showing | Check `static/` folder exists |
| Prediction fails | Verify all 20 fields have values |
| Word report error | `pip install python-docx` |
| Numpy version error | `pip install numpy<2.0` |

---

## Model Training (if needed)

**⚠️ Only run if retraining:**
```bash
python model.py
```

This will:
1. Load ECG-Dataset.csv
2. Split data (80% train, 20% test)
3. Train neural network for 50 epochs
4. Save: `heart_disease_model.h5` + `scaler.save`

---

## Key Python Dependencies

```python
flask              # Web framework
tensorflow/keras   # Neural network
pandas/numpy       # Data processing
scikit-learn       # Data scaling
matplotlib/seaborn # Visualization
python-docx        # Word document generation
```

---

## Customization Examples

### Change prediction threshold (stricter)
```python
# In app.py, line ~70
prediction = 1 if prediction_prob_value > 0.6 else 0  # was 0.5
```

### Add more layers to model
```python
# In model.py, line ~22
model = tf.keras.Sequential([
    Dense(128, activation='relu', input_shape=(20,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),  # New layer
    Dense(1, activation='sigmoid')
])
```

### Change output graphs
```python
# In app.py, generate_visualizations() function
# Modify colors, add more plots, change titles
```

---

## Testing the Model

```python
# Test prediction manually
python
>>> import numpy as np
>>> from tensorflow.keras.models import load_model
>>> import joblib
>>> model = load_model('heart_disease_model.h5')
>>> scaler = joblib.load('scaler.save')
>>> test_data = np.array([[55, 1, 0, 0, 113, ...]])  # 20 values
>>> scaled = scaler.transform(test_data)
>>> prediction = model.predict(scaled)
>>> print(prediction[0][0])  # Probability
0.35
```

---

## Performance Metrics

- **Model Accuracy:** ~85-90%
- **Prediction Speed:** <100ms per patient
- **Training Time:** 2-5 minutes on CPU
- **Model Size:** ~150KB
- **Scaler Size:** ~3KB

---

## Security Notes

✓ All inputs validated  
✓ Old files auto-cleaned  
✓ No data persistence  
✓ StandardScaler ensures consistency  
✓ Input range validation  

⚠️ For production: add HTTPS, rate limiting, authentication

---

## Deployment Options

### Local Development
```bash
python app.py
```

### Production (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker
```bash
docker build -t heart-disease-app .
docker run -p 5000:5000 heart-disease-app
```

---

## Documentation Files

- **README.md** - Project overview & features
- **DEVELOPER_GUIDE.md** - Complete technical documentation
- **requirements.txt** - Python package versions
- **QUICK_REFERENCE.md** - This file

---

**Need help?** Check DEVELOPER_GUIDE.md for detailed explanations.

**Status:** ✅ Ready to Use  
**Last Updated:** January 29, 2026
