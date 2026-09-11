import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense


# Load processed training data
train_data = pd.read_csv("data/processed_train.csv")

# Load test data separately
test_data = pd.read_csv("data/processed_test.csv")


# Get temperature values
train_temperature = train_data["meantemp"].values.reshape(-1, 1)
test_temperature = test_data["meantemp"].values.reshape(-1, 1)


# Scale data
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

X_train, y_train = create_sequences(
    train_scaled,
    sequence_length
)

X_test, y_test = create_sequences(
    test_scaled,
    sequence_length
)


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# Build RNN model
model = Sequential([
    Input(shape=(7, 1)),
    SimpleRNN(32, activation="tanh"),
    Dense(1)
])


# Compile model
model.compile(
    optimizer="adam",
    loss="mse"
)


# Display model architecture
model.summary()


# Train model
# validation_split uses only training data for validation
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# Save trained model
model.save("models/temperature_rnn.keras")

print("\nModel trained and saved successfully!")