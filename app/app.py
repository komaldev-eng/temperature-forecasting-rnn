from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load training data
train_data = pd.read_csv("data/processed_train.csv")

# Get temperature values
train_temperature = train_data["meantemp"].values.reshape(-1, 1)

# Create scaler using training data
scaler = MinMaxScaler()
scaler.fit(train_temperature)

# Load trained RNN model
model = load_model("models/temperature_rnn.keras")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    temperatures = data.get("temperatures")

    # Check whether exactly 7 temperatures were provided
    if not temperatures or len(temperatures) != 7:
        return jsonify({
            "error": "Please provide exactly 7 temperatures."
        }), 400

    # Convert input to NumPy array
    temperatures = np.array(temperatures).reshape(-1, 1)

    # Scale input
    temperatures_scaled = scaler.transform(temperatures)

    # Reshape for RNN
    input_data = temperatures_scaled.reshape(1, 7, 1)

    # Make prediction
    prediction_scaled = model.predict(input_data, verbose=0)

    # Convert back to Celsius
    prediction = scaler.inverse_transform(prediction_scaled)

    return jsonify({
        "predicted_temperature": round(float(prediction[0][0]), 2)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)