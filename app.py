from flask import Flask, render_template, request, flash
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for flash messages

# Load your best model (example)
model = pickle.load(open('best_model.pkl', 'rb'))

@app.route('/')
def index():
    labels = [
        'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI',
        'Diabetes Pedigree Function', 'Age', 'Daily Steps', 'Sleep Hours', 'Diet Score (1-10)'
    ]
    names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age', 'daily_steps', 'sleep_hours', 'diet_score'
    ]
    zipped = zip(labels, names)
    return render_template('index.html', zipped=zipped)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data and convert to float list
        features = [float(request.form[name]) for name in [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
            'DiabetesPedigreeFunction', 'Age', 'daily_steps', 'sleep_hours', 'diet_score'
        ]]
        
        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)[0]
        proba = model.predict_proba(final_features)[0][prediction]
        
        if prediction == 1:
            result = "The person is likely to have Diabetes."
        else:
            result = "The person is unlikely to have Diabetes."
            
        confidence = f"Prediction confidence: {proba * 100:.2f}%"
        
        # Pass zipped again for re-rendering form
        labels = [
            'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI',
            'Diabetes Pedigree Function', 'Age', 'Daily Steps', 'Sleep Hours', 'Diet Score (1-10)'
        ]
        names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
            'DiabetesPedigreeFunction', 'Age', 'daily_steps', 'sleep_hours', 'diet_score'
        ]
        zipped = zip(labels, names)
        
        return render_template('index.html', prediction_text=result, confidence_text=confidence, zipped=zipped)
    
    except Exception as e:
        flash(f"Error occurred: {str(e)}", 'danger')
        return render_template('index.html', zipped=zip(
            [
                'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI',
                'Diabetes Pedigree Function', 'Age', 'Daily Steps', 'Sleep Hours', 'Diet Score (1-10)'
            ],
            [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age', 'daily_steps', 'sleep_hours', 'diet_score'
            ]
        ))

if __name__ == "__main__":
    app.run(debug=True)
