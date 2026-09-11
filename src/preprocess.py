import pandas as pd

# Load training and testing data
train_data = pd.read_csv("data/DailyDelhiClimateTrain.csv")
test_data = pd.read_csv("data/DailyDelhiClimateTest.csv")

# Convert date column to datetime
train_data["date"] = pd.to_datetime(train_data["date"])
test_data["date"] = pd.to_datetime(test_data["date"])

# Sort data by date
train_data = train_data.sort_values("date")
test_data = test_data.sort_values("date")

# Check missing values
print("Missing values in training data:")
print(train_data.isnull().sum())

print("\nMissing values in testing data:")
print(test_data.isnull().sum())

# Keep only date and mean temperature
train_processed = train_data[["date", "meantemp"]].copy()
test_processed = test_data[["date", "meantemp"]].copy()

# Reset index
train_processed = train_processed.reset_index(drop=True)
test_processed = test_processed.reset_index(drop=True)

# Display processed data
print("\nProcessed training data:")
print(train_processed.head())

print("\nProcessed testing data:")
print(test_processed.head())

print("\nTraining records:", len(train_processed))
print("Testing records:", len(test_processed))

# Save processed datasets
train_processed.to_csv(
    "data/processed_train.csv",
    index=False
)

test_processed.to_csv(
    "data/processed_test.csv",
    index=False
)

print("\nProcessed datasets saved successfully!")