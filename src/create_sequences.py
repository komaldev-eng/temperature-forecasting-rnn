import pandas as pd
import numpy as np

# Load processed training and testing data
train_data = pd.read_csv("data/processed_train.csv")
test_data = pd.read_csv("data/processed_test.csv")

# Get temperature values
train_temperature = train_data["meantemp"].values
test_temperature = test_data["meantemp"].values


# Function to create sequences
def create_sequences(data, sequence_length):
    X = []
    y = []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])

    return np.array(X), np.array(y)


# Number of previous days used for prediction
sequence_length = 7

# Create training sequences
X_train, y_train = create_sequences(
    train_temperature,
    sequence_length
)

# Create testing sequences
X_test, y_test = create_sequences(
    test_temperature,
    sequence_length
)

# Display results
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\nFirst training sequence:")
print(X_train[0])

print("\nTarget for first sequence:")
print(y_train[0])