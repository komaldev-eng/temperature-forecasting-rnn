import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


# Load processed data
train_data = pd.read_csv("data/processed_train.csv")
test_data = pd.read_csv("data/processed_test.csv")


# Get temperature values
train_temperature = train_data["meantemp"].values.reshape(-1, 1)
test_temperature = test_data["meantemp"].values.reshape(-1, 1)


# Scale data using the same approach as training
scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_temperature)
test_scaled = scaler.transform(test_temperature)


# Create sequences
def create_sequences(data, sequence_length):
    X = []
    y = []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])

    return np.array(X), np.array(y)


# Use previous 7 days to predict the next day
sequence_length = 7

X_test, y_test = create_sequences(
    test_scaled,
    sequence_length
)


print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# Load trained RNN model
model = load_model("models/temperature_rnn.keras")


# Make predictions
predictions_scaled = model.predict(X_test)


# Convert scaled values back to Celsius
predictions = scaler.inverse_transform(predictions_scaled)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))


# Calculate MAE
mae = mean_absolute_error(actual, predictions)


# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual, predictions))


print("\nModel Evaluation")
print("----------------")
print(f"MAE: {mae:.2f} °C")
print(f"RMSE: {rmse:.2f} °C")


# Display actual vs predicted values
print("\nActual vs Predicted:")
print("--------------------")

for i in range(10):
    print(
        f"Actual: {actual[i][0]:.2f} °C | "
        f"Predicted: {predictions[i][0]:.2f} °C"
    )


# Plot actual vs predicted temperatures
plt.figure(figsize=(12, 5))

plt.plot(actual, label="Actual Temperature")
plt.plot(predictions, label="Predicted Temperature")

plt.title("Actual vs Predicted Temperature")
plt.xlabel("Test Day")
plt.ylabel("Temperature (°C)")

plt.legend()
plt.tight_layout()

plt.savefig("plots/actual_vs_predicted.png")

plt.show()

print("\nPlot saved successfully!")