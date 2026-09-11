import pandas as pd

# Load training data
train_data = pd.read_csv("data/DailyDelhiClimateTrain.csv")

# Load testing data
test_data = pd.read_csv("data/DailyDelhiClimateTest.csv")

print("TRAINING DATA")
print(train_data.head())

print("\nTraining data shape:")
print(train_data.shape)

print("\nTraining data columns:")
print(train_data.columns)

print("\nTraining data types:")
print(train_data.dtypes)

print("\nTraining data missing values:")
print(train_data.isnull().sum())


print("\n" + "=" * 50)
print("TESTING DATA")
print("=" * 50)

print(test_data.head())

print("\nTesting data shape:")
print(test_data.shape)

print("\nTesting data columns:")
print(test_data.columns)

print("\nTesting data types:")
print(test_data.dtypes)

print("\nTesting data missing values:")
print(test_data.isnull().sum())