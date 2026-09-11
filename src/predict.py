import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


# Load training data
train_data = pd.read_csv("data/processed_train.csv")

# Get training temperatures
train_temperature = train_data["meantemp"].values.reshape(-1, 1)


# Create and fit scaler
scaler = MinMaxScaler()
scaler.fit(train_temperature)


# Load trained model
model = load_model("models/temperature_rnn.keras")


# Take 7 temperatures from the user
print("Enter the temperature for the last 7 days:")

temperatures = []

for i in range(7):
    temp = float(input(f"Day {i + 1}: "))
    temperatures.append(temp)


# Convert to NumPy array
temperatures = np.array(temperatures).reshape(-1, 1)


# Scale the temperatures
temperatures_scaled = scaler.transform(temperatures)


# Reshape for RNN
input_data = temperatures_scaled.reshape(1, 7, 1)


# Make prediction
prediction_scaled = model.predict(input_data)


# Convert prediction back to Celsius
prediction = scaler.inverse_transform(prediction_scaled)


# Display result
print("\n-----------------------------")
print("Temperature Forecast")
print("-----------------------------")
print(f"Predicted next-day temperature: {prediction[0][0]:.2f} °C")