from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from keras.models import load_model
from keras.utils.np_utils import to_categorical

app = Flask(__name__)


# Load the model
model = load_model('F:/main project/model.h5')


# Load dataset
data = pd.read_csv('ECG-Dataset.csv')
data.columns = ['age','sex','smoker','years_of_smoking','LDL_cholesterol','chest_pain_type','height','weight', 'familyhist',
                'activity', 'lifestyle', 'cardiac intervention', 'heart_rate', 'diabetes', 'blood_pressure_sys', 'blood_pressure_dias', 
                 'hypertension', 'interventricular_septal_end_diastole', 'ecg_pattern', 'Q_wave', 'target']

# Preprocessing
X = data.drop('target', axis=1)
y = data['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = [float(request.form[f'input_{i+1}']) for i in range(X_scaled.shape[1])]
        user_input_scaled = scaler.transform([user_input])

        # Make prediction
        prediction = model.predict(user_input_scaled)
        if prediction >= 0.5:
            prediction_text = "Heart disease present"
        else:
            prediction_text = "No heart disease"

        # Create plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        sns.histplot(data['age'], ax=axes[0, 0])
        sns.countplot(x='sex', data=data, ax=axes[0, 1])
        sns.heatmap(data.corr(), annot=True, fmt='.1f', ax=axes[1, 0])
        pd.crosstab(data['age'], data['target']).plot(kind="bar", figsize=(15, 5), ax=axes[1, 1])
        axes[1, 1].set_title('Heart Disease Frequency for Ages')
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend(['No Disease', 'Disease'])
        axes[2, 0].text(0.5, 0.5, prediction_text, horizontalalignment='center', verticalalignment='center', fontsize=20)
        axes[2, 0].axis('off')
        axes[2, 1].axis('off')
        
        # Save plots to a temporary file
        temp_file = 'static/plots.png'
        plt.savefig(temp_file)
        plt.close()

        return render_template('result.html', prediction=prediction_text, plot=temp_file)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


