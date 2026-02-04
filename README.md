# ❤️ Heart Disease Prediction System

A machine learning web application that predicts heart disease risk using Deep Neural Networks.

## ✨ Features

- **20-parameter medical assessment** - Comprehensive patient evaluation
- **Deep Neural Network AI** - 3-layer neural network trained on ECG dataset
- **Interactive web interface** - User-friendly form with helpful descriptions
- **Real-time predictions** - <100ms prediction time
- **Risk visualization** - 3 professional graphs and charts
- **Word report generation** - Auto-generate downloadable medical reports
- **Responsive design** - Works on desktop and tablet

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Flask
- TensorFlow
- Pandas, NumPy, Scikit-learn

### Installation
```bash
cd "c:\Users\tharu\Downloads\main project"
pip install -r requirements.txt
python app.py
```

Open browser: `http://127.0.0.1:5000`

## 📊 Model Architecture

```
Input (20 features)
    ↓
Dense Layer 1: 64 neurons + ReLU
    ↓
Dense Layer 2: 32 neurons + ReLU
    ↓
Output: 1 neuron + Sigmoid
    ↓
Prediction: Probability (0-1)
```

**Algorithm:** Deep Neural Network (ANN)  
**Framework:** TensorFlow/Keras  
**Training:** Binary Crossentropy Loss, Adam Optimizer  

## 📁 Project Structure

```
├── app.py                 # Main Flask application
├── model.py              # Model training script
├── DEVELOPER_GUIDE.md    # Detailed technical documentation
├── templates/
│   ├── index.html        # Input form
│   └── result.html       # Results page
├── static/               # Generated graphs
├── ECG-Dataset.csv       # Training data
├── heart_disease_model.h5 # Trained model
└── scaler.save          # Feature scaler
```

## 💡 How It Works

1. **Input:** User enters 20 medical parameters through web form
2. **Preprocessing:** Features are standardized using saved scaler
3. **Prediction:** Deep neural network processes features
4. **Output:** Probability score (0-1) converted to prediction + confidence
5. **Visualization:** 3 graphs generated showing patient metrics and correlations
6. **Report:** Optional Word document download with all results

## 🔍 Input Features

- Demographics: Age, Sex
- Lifestyle: Smoking status, Years smoking, Physical activity
- Clinical: Chest pain type, ECG pattern, Q wave
- Vitals: Heart rate, Blood pressure (systolic/diastolic)
- Medical: LDL cholesterol, Height, Weight, Diabetes, Hypertension
- Assessment: Family history, Previous interventions

## 📈 Prediction Output

**Example Result:**
```
✓ NEGATIVE: Patient likely does NOT have heart disease
Confidence: 78%
Risk Level: LOW RISK

Patient Metrics:
- Age: 55 years
- Heart Rate: 84 bpm
- Blood Pressure: 124/75 mmHg
- BMI: 31.2
- Risk Factors: None identified
```

## 🎯 Use Cases

- **Healthcare Professionals** - Quick screening tool for patient assessment
- **Telemedicine** - Remote cardiac risk assessment
- **Research** - Medical data analysis and validation
- **Education** - Teaching AI/ML applications in healthcare
- **Recruitment** - Demo project for ML engineer interviews

## 🔧 Developer Resources

See **DEVELOPER_GUIDE.md** for:
- Detailed algorithm explanation
- Architecture overview
- Installation instructions
- API documentation
- Customization guide
- Troubleshooting

## ⚙️ Technical Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask |
| ML Framework | TensorFlow/Keras |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Reports | Python-docx |
| Frontend | HTML5, CSS3, JavaScript |

## 📊 Dataset

- **Size:** 335 patient records
- **Features:** 20 medical parameters
- **Target:** Binary (Heart disease: Yes/No)
- **Source:** ECG-Dataset.csv

## ⚠️ Important Disclaimer

This application is provided for **educational and research purposes only**. It should NOT be used for:
- Actual medical diagnosis
- Treatment decisions
- Clinical deployment without proper validation
- Replacing professional medical advice

Always consult qualified healthcare professionals for medical decisions.

## 🔒 Security

- Input validation on all fields
- Automatic cleanup of generated files
- No patient data persistence
- StandardScaler ensures consistent feature scaling

## 🚀 Deployment

### Development
```bash
python app.py  # Runs on http://127.0.0.1:5000
```

### Production
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📞 Support

For detailed technical information, refer to `DEVELOPER_GUIDE.md`

## 📄 License

Educational use - See LICENSE file

---

**Status:** ✅ Production Ready  
**Version:** 1.0  
**Last Updated:** January 29, 2026
