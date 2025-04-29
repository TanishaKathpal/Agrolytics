from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'supersecret'

# Load models
crop_model = joblib.load('final_model_recommendation.sav')
yield_model = joblib.load('final_model_yield_prediction.sav')

# Label maps for Crop Recommendation
crop_label_map = {
    20: "Rice",
    11: "Maize",
    3: "ChickPea",
    9: "KidneyBeans",
    18: "PigeonPeas",
    13: "MothBeans",
    14: "MungBean",
    2: "Blackgram",
    10: "Lentil",
    19: "Pomegranate",
    1: "Banana",
    12: "Mango",
    7: "Grapes",
    21: "Watermelon",
    15: "Muskmelon",
    0: "Apple",
    16: "Orange",
    17: "Papaya",
    4: "Coconut",
    6: "Cotton",
    8: "Jute",
    5: "Coffee"
}

# Area and Item label maps for Yield Prediction
area_label_map = {
    0: 'Albania', 1: 'Algeria', 2: 'Angola', 3: 'Argentina', 4: 'Armenia', 5: 'Australia', 6: 'Austria', 7: 'Azerbaijan',
    8: 'Bahamas', 9: 'Bahrain', 10: 'Bangladesh', 11: 'Belarus', 12: 'Belgium', 13: 'Botswana', 14: 'Brazil', 15: 'Bulgaria',
    16: 'Burkina Faso', 17: 'Burundi', 18: 'Cameroon', 19: 'Canada', 20: 'Central African Republic', 21: 'Chile', 22: 'Colombia',
    23: 'Croatia', 24: 'Denmark', 25: 'Dominican Republic', 26: 'Ecuador', 27: 'Egypt', 28: 'El Salvador', 29: 'Eritrea',
    30: 'Estonia', 31: 'Finland', 32: 'France', 33: 'Germany', 34: 'Ghana', 35: 'Greece', 36: 'Guatemala', 37: 'Guinea',
    38: 'Guyana', 39: 'Haiti', 40: 'Honduras', 41: 'Hungary', 42: 'India', 43: 'Indonesia', 44: 'Iraq', 45: 'Ireland',
    46: 'Italy', 47: 'Jamaica', 48: 'Japan', 49: 'Kazakhstan', 50: 'Kenya', 51: 'Latvia', 52: 'Lebanon', 53: 'Lesotho',
    54: 'Libya', 55: 'Lithuania', 56: 'Madagascar', 57: 'Malawi', 58: 'Malaysia', 59: 'Mali', 60: 'Mauritania', 61: 'Mauritius',
    62: 'Mexico', 63: 'Montenegro', 64: 'Morocco', 65: 'Mozambique', 66: 'Namibia', 67: 'Nepal', 68: 'Netherlands',
    69: 'New Zealand', 70: 'Nicaragua', 71: 'Niger', 72: 'Norway', 73: 'Pakistan', 74: 'Papua New Guinea', 75: 'Peru',
    76: 'Poland', 77: 'Portugal', 78: 'Qatar', 79: 'Romania', 80: 'Rwanda', 81: 'Saudi Arabia', 82: 'Senegal', 83: 'Slovenia',
    84: 'South Africa', 85: 'Spain', 86: 'Sri Lanka', 87: 'Sudan', 88: 'Suriname', 89: 'Sweden', 90: 'Switzerland',
    91: 'Tajikistan', 92: 'Thailand', 93: 'Tunisia', 94: 'Turkey', 95: 'Uganda', 96: 'Ukraine', 97: 'United Kingdom',
    98: 'Uruguay', 99: 'Zambia', 100: 'Zimbabwe'
}

item_label_map = {
    0: 'Cassava', 1: 'Maize', 2: 'Plantains and others', 3: 'Potatoes',
    4: 'Rice, paddy', 5: 'Sorghum', 6: 'Soybeans', 7: 'Sweet potatoes',
    8: 'Wheat', 9: 'Yams'
}

# Routes

@app.route('/')
def index():
    return render_template('index.html')

# 🌱 Crop Recommendation

@app.route('/crop_recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if request.method == 'POST':
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        features = [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]
        print("🧪 Crop Model Input Features:", features)

        crop_prediction = crop_model.predict(features)
        print("🧪 Crop Prediction Output (Encoded):", crop_prediction)

        crop_encoded = int(crop_prediction[0])
        session['crop_encoded'] = crop_encoded
        return redirect(url_for('crop_result'))

    return render_template('crop_recommendation.html')

@app.route('/crop_result')
def crop_result():
    crop_encoded = session.get('crop_encoded', -1)
    prediction_label = crop_label_map.get(crop_encoded, "Unknown")
    return render_template('result_crop.html', prediction=crop_encoded, prediction_label=prediction_label)

# 🌾 Yield Prediction
@app.route('/yield_prediction', methods=['GET', 'POST'])
def yield_prediction():
    if request.method == 'POST':
        area = int(request.form['area'])
        item = int(request.form['item'])
        average_rainfall = float(request.form['average_rainfall'])
        pesticides = float(request.form['pesticides'])

        features = [[area, item, average_rainfall, pesticides]]
        print("🧪 Yield Model Input Features:", features)

        yield_prediction = yield_model.predict(features)
        print("🧪 Yield Prediction Output:", yield_prediction)

        yield_predicted = yield_prediction[0]
        session['yield_prediction'] = yield_predicted
        return redirect(url_for('yield_result'))

    return render_template('yield_prediction.html', area_label_map=area_label_map, item_label_map=item_label_map)

@app.route('/yield_result')
def yield_result():
    yield_predicted = session.get('yield_prediction', None)
    return render_template('result_yield.html', prediction=yield_predicted)

# Run
if __name__ == '__main__':
    app.run(debug=True)
