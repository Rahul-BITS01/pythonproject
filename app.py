import json
import pickle
import warnings

from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model and scaler
regmodel = pickle.load(open('ml_model.pkl', 'rb'))
scalar = pickle.load(open('scaler_model.pkl', 'rb'))

# Suppress warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text="The Anomaly Detection Class Prediction is [ Anomaly : 0 , Normal : 1] : {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)
