from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# 1. Load the trained model
# Ensure 'crop_model.pkl' exists in the same folder
try:
    model = joblib.load('crop_model.pkl')
except FileNotFoundError:
    print("Error: 'crop_model.pkl' not found. Please run train_model.py first.")
    model = None

# 2. Mock data for Mandi Rates and Govt Schemes
# In a real-world app, these would come from an API or Database
MANDI_DATA = [
    {'crop': 'Rice', 'market': 'Kolkata', 'price': '₹2,100', 'trend': 'up'},
    {'crop': 'Wheat', 'market': 'Ludhiana', 'price': '₹2,015', 'trend': 'down'},
    {'crop': 'Cotton', 'market': 'Nagpur', 'price': '₹6,500', 'trend': 'up'},
]

SCHEMES_DATA = [
    {'name': 'PM-Kisan Samman Nidhi', 'benefit': '₹6,000 annual income support for farmers.', 'link': 'https://pmkisan.gov.in/'},
    {'name': 'Pradhan Mantri Fasal Bima Yojana', 'benefit': 'Crop insurance against natural calamities.', 'link': 'https://pmfby.gov.in/'},
]

@app.route('/')
def index():
    # Renders the main page with Mandi and Schemes data pre-loaded
    return render_template('index.html', prices=MANDI_DATA, schemes=SCHEMES_DATA)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Error: Model not loaded.", prices=MANDI_DATA, schemes=SCHEMES_DATA)

    try:
        # Extract data from form
        n = float(request.form['n'])
        p = float(request.form['p'])
        k = float(request.form['k'])
        temp = float(request.form['t'])
        hum = float(request.form['h'])
        
        # Note: Your model was trained on 7 features (N, P, K, temp, hum, ph, rainfall)
        # We will provide default values for pH (6.5) and Rainfall (100) since they aren't in your HTML yet
        features = [[n, p, k, temp, hum, 6.5, 100.0]]
        
        prediction = model.predict(features)
        result = f"The best crop for your soil is: {prediction[0].upper()}"

    except Exception as e:
        result = f"Error in prediction: {str(e)}"

    return render_template('index.html', 
                           prediction_text=result, 
                           prices=MANDI_DATA, 
                           schemes=SCHEMES_DATA)

if __name__ == "__main__":
    # Required for 24/7 cloud hosting (dynamic port assignment)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)