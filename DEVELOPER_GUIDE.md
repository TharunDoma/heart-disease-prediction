# Heart Disease Prediction System - Developer Guide

## 📋 Project Overview

A web-based machine learning application that predicts heart disease risk using a Deep Neural Network (DNN). The system provides an interactive interface for healthcare professionals and researchers to assess patient cardiac health based on 20 medical parameters.

**Technology Stack:**
- **Backend:** Flask (Python)
- **ML Model:** TensorFlow/Keras Deep Neural Network
- **Frontend:** HTML5, CSS3, JavaScript
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Report Generation:** Python-docx

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Browser (Frontend)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  HTML Form + JavaScript                              │   │
│  │  - 20 medical input fields                            │   │
│  │  - Dropdown menus for categorical data                │   │
│  │  - Sample data button for quick demo                  │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP Request (POST)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Flask Web Server (Backend)                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Routes:                                               │   │
│  │ - GET /         → Load form                           │   │
│  │ - POST /predict → Process & predict                   │   │
│  │ - POST /download-report → Generate Word document      │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Data Processing & ML Pipeline                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. Input Validation (20 features)                     │   │
│  │ 2. Feature Scaling (StandardScaler)                   │   │
│  │ 3. Neural Network Prediction                          │   │
│  │ 4. Visualization Generation                           │   │
│  │ 5. Report Assembly                                    │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP Response + Images
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Result Page + Download Options                  │
│  - Prediction result with confidence score                   │
│  - Patient health profile                                    │
│  - 3 visualization graphs                                    │
│  - Word document download link                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Machine Learning Algorithm

### Algorithm Type: Deep Neural Network (ANN)

**Model Architecture:**
```
Input Layer (20 features)
    ↓
Dense(64, activation='relu')  [ReLU introduces non-linearity]
    ↓
Dense(32, activation='relu')  [Further feature refinement]
    ↓
Dense(1, activation='sigmoid') [Binary classification output]
    ↓
Output: Probability (0.0 - 1.0)
```

**Training Configuration:**
```python
Optimizer:     Adam (adaptive learning rate)
Loss Function: Binary Crossentropy (for binary classification)
Metric:        Accuracy
Epochs:        50
Train/Test:    80/20 split
Validation:    20% of training data
```

**Prediction Pipeline:**
```
1. User Input → 20 features collected
2. Standardization → Scale to mean=0, std=1 using saved scaler
3. Neural Network → Forward pass through trained model
4. Sigmoid Output → Probability between 0-1
5. Threshold → If prob > 0.5 → Positive (disease), else Negative
6. Confidence → abs(prob - 0.5) × 2 × 100 = confidence %
```

**Input Features (20 total):**
| # | Feature | Range | Type |
|---|---------|-------|------|
| 1 | age | 20-90 | Numeric |
| 2 | sex | 0-1 | Binary (0=Female, 1=Male) |
| 3 | smoke | 0-1 | Binary |
| 4 | years | 0-50 | Numeric |
| 5 | ldl | 26-260 | Numeric (mg/dL) |
| 6 | chp | 1-4 | Categorical (chest pain type) |
| 7 | height | 128-192 | Numeric (cm) |
| 8 | weight | 41-134 | Numeric (kg) |
| 9 | fh | 0-1 | Binary (family history) |
| 10 | active | 0-1 | Binary |
| 11 | lifestyle | 1-3 | Categorical |
| 12 | ihd | 0-1 | Binary |
| 13 | hr | 40-140 | Numeric (bpm) |
| 14 | dm | 0-1 | Binary (diabetes) |
| 15 | bpsys | 80-220 | Numeric (mmHg) |
| 16 | bpdias | 40-140 | Numeric (mmHg) |
| 17 | htn | 0-1 | Binary (hypertension) |
| 18 | ivsd | 0-1 | Binary |
| 19 | ecgpatt | 1-4 | Categorical |
| 20 | qwave | 0-1 | Binary |

---

## 📁 Project Structure

```
main project/
├── app.py                          # Main Flask application
├── model.py                        # Model training script
├── DEVELOPER_GUIDE.md              # This file
├── ECG-Dataset.csv                 # Training dataset (335 samples)
├── heart_disease_model.h5          # Trained neural network model
├── scaler.save                     # Fitted StandardScaler
├── templates/
│   ├── index.html                  # Input form page
│   └── result.html                 # Results & visualization page
├── static/
│   ├── patient_metrics.png         # Generated dashboard graph
│   ├── correlation_heatmap.png     # Feature correlation heatmap
│   └── risk_distribution.png       # Dataset distribution graph
└── reports/                        # (Auto-created) Generated Word documents
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.12+
- pip (Python package manager)
- Anaconda (recommended for dependency management)

### Step 1: Clone/Download Project
```bash
cd c:\Users\tharu\Downloads\main project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Or install individually:**
```bash
pip install flask numpy pandas tensorflow scikit-learn matplotlib seaborn joblib python-docx
```

### Step 3: Verify Installation
```bash
python -c "import tensorflow; print('TensorFlow OK')"
python -c "import flask; print('Flask OK')"
```

### Step 4: Run the Application
```bash
python app.py
```

**Output:**
```
* Running on http://127.0.0.1:5000
INFO:werkzeug:Press CTRL+C to quit
```

### Step 5: Access in Browser
Open: `http://127.0.0.1:5000`

---

## 📚 Key Files Explained

### `app.py` - Main Application
**Routes:**
- `GET /` - Renders index.html with input form
- `POST /predict` - Processes form data, makes prediction, generates visualizations
- `POST /download-report` - Generates and downloads Word document

**Key Functions:**
```python
clean_old_plots()           # Deletes old PNG files before new predictions
generate_visualizations()   # Creates 3 graphs
predict()                   # Main prediction route
download_report()           # Word document generation
```

**Flow:**
```
1. Receive 20 form fields
2. Convert to float values
3. Scale using saved scaler
4. Predict using trained model
5. Calculate confidence
6. Generate visualizations
7. Return result.html with data
```

### `model.py` - Model Training
**One-time execution script to train the neural network:**
```python
1. Load ECG-Dataset.csv
2. Split: 80% train, 20% test
3. Standardize features with StandardScaler
4. Build Sequential model (3 layers)
5. Compile with Adam optimizer
6. Train for 50 epochs
7. Save model → heart_disease_model.h5
8. Save scaler → scaler.save
```

**Do NOT run again unless retraining:**
```bash
# Only if you want to retrain with new data
python model.py
```

### `templates/index.html` - Input Form
**Features:**
- 20 form fields with validation
- Dropdown menus for categorical data
- Helpful descriptions for each field
- "Fill Sample Data" button for quick testing
- Responsive design

**Field Groups:**
- Demographics: Age, Sex
- Lifestyle: Smoker, Years, Physical Activity
- Medical: LDL, Heart Rate, Blood Pressure, Diabetes, Hypertension
- Physical: Height, Weight, BMI
- Clinical: Chest Pain, ECG Pattern, Q Wave

### `templates/result.html` - Results Page
**Displays:**
- Prediction result (POSITIVE/NEGATIVE)
- Confidence percentage with progress bar
- Patient health profile (10 metrics)
- Risk factors summary
- 3 visualization graphs
- Download Word report button
- Print button

### `ECG-Dataset.csv` - Training Data
**Format:**
```
age,sex,smoke,years,ldl,chp,height,weight,fh,active,lifestyle,ihd,hr,dm,bpsys,bpdias,htn,ivsd,ecgpatt,qwave,target
65,0,0,0,69,4,168,111,1,0,1,1,98,1,120,80,1,0,4,0,0
54,1,0,0,117,2,145,81,0,0,2,0,85,0,130,80,0,0,4,0,0
...
```
- 335 total samples
- Last column = target (0=no disease, 1=has disease)
- 20 input features, 1 output

---

## 🔄 Data Flow Example

### User Submits Form:
```
Input: 55-year-old male, LDL=113, HR=84, BP=124/75, etc.
    ↓
[20 values collected from form fields]
    ↓
[Features: [55, 1, 0, 0, 113, 2, 162, 82, ...]]
```

### Processing:
```
Raw Input
    ↓
StandardScaler.transform()
[Normalize: (value - mean) / std]
    ↓
Scaled Features
[[-0.45, 0.89, -0.12, ..., 0.67]]
```

### Prediction:
```
Neural Network Forward Pass
Input → Dense(64) → ReLU → Dense(32) → ReLU → Dense(1) → Sigmoid
    ↓
Output: 0.35 (35% probability of disease)
    ↓
Prediction: 0 (Negative - No Disease)
Confidence: (1 - 0.35) × 100 = 65%
```

### Visualization:
```
1. Patient Metrics Dashboard
   - 6 bar charts with color-coded risk levels
   - Risk factors summary box

2. Feature Correlation Heatmap
   - Shows relationships between key health metrics
   - Correlation with target variable

3. Risk Distribution
   - Bar chart of dataset (healthy vs diseased patients)
   - Percentage breakdown
```

### Report Generation:
```
Word Document Creation:
├── Title & Timestamp
├── Prediction Result
├── Patient Profile Table
├── Risk Factors List
├── 3 Embedded Graphs
├── Disclaimer
└── Save as .docx file
```

---

## 🔧 Configuration & Customization

### Change Model Architecture
Edit `model.py`:
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(20,)),  # 128 instead of 64
    tf.keras.layers.Dropout(0.2),                                      # Add dropout
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Change Prediction Threshold
Edit `app.py` (currently 0.5):
```python
prediction = 1 if prediction_prob_value > 0.6 else 0  # More conservative
```

### Add New Features
1. Update HTML form (index.html)
2. Update feature count (currently 20)
3. Retrain model with new dataset
4. Update patient_profile dict in app.py

### Modify Visualizations
Edit `generate_visualizations()` in app.py:
```python
# Change colors
color = '#e74c3c'  # Red
# Change axes labels
ax.set_xlabel('Custom Label')
# Add more plots
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **TensorFlow import error** | Downgrade numpy: `pip install numpy<2.0` |
| **Port 5000 already in use** | Change port in app.py: `app.run(port=5001)` |
| **Model not found** | Ensure `heart_disease_model.h5` exists in project root |
| **Scaler mismatch** | Retrain with: `python model.py` |
| **Graphs not generating** | Check `static/` directory exists |
| **Word report fails** | Verify `python-docx` installed: `pip install python-docx` |

---

## 📊 Model Performance Notes

- **Accuracy:** ~85-90% (varies with dataset)
- **Precision:** Optimized for balanced sensitivity/specificity
- **Training Time:** ~2-5 minutes on CPU
- **Prediction Time:** <100ms per patient
- **Model Size:** ~150KB

---

## 🔒 Security Considerations

1. **Input Validation:** All inputs converted to float, validated for range
2. **File Cleanup:** Old PNG files deleted before new predictions
3. **No Data Storage:** Patient data not persisted (except in reports)
4. **HTTPS:** Recommended for production deployment
5. **Rate Limiting:** Not implemented (add for production)

---

## 📱 Deployment

### For Production:
```bash
# Use WSGI server instead of Flask's development server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment:
```dockerfile
FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

---

## 🧪 Testing

### Test Prediction Manually:
```python
python
>>> import numpy as np
>>> from tensorflow.keras.models import load_model
>>> import joblib
>>> 
>>> model = load_model('heart_disease_model.h5')
>>> scaler = joblib.load('scaler.save')
>>> 
>>> # Test with sample data (20 features)
>>> test_data = np.array([[55, 1, 0, 0, 113, 2, 162, 82, 0, 1, 2, 0, 84, 0, 124, 75, 0, 0, 4, 0]])
>>> scaled = scaler.transform(test_data)
>>> prediction = model.predict(scaled)
>>> print(f"Probability: {prediction[0][0]:.2f}")
```

### Test Flask App:
```bash
python -m pytest test_app.py  # If test file exists
```

---

## 📝 Development Tips

1. **Use virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Keep dependencies updated:**
   ```bash
   pip list --outdated
   ```

3. **Monitor model performance:**
   - Track accuracy metrics over time
   - Retrain periodically with new data

4. **Log predictions:**
   - Add logging to app.py for audit trail
   - Store prediction history for analysis

5. **Version control:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

---

## 📖 References

- **TensorFlow:** https://www.tensorflow.org/
- **Flask:** https://flask.palletsprojects.com/
- **Scikit-learn:** https://scikit-learn.org/
- **Neural Networks:** https://cs231n.github.io/

---

## 👨‍💻 Support & Contribution

For issues, questions, or improvements:
1. Review this guide
2. Check troubleshooting section
3. Review code comments in app.py
4. Test with sample data first

---

**Last Updated:** January 29, 2026  
**Version:** 1.0  
**Status:** Production Ready ✅
