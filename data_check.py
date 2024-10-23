import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load train.parquet and test.parquet
train_data = pd.read_parquet('train.parquet')
test_data = pd.read_parquet('test.parquet')

# Display basic info about the datasets
print("Training Data Info:")
print(train_data.info())

print("\nTest Data Info:")
print(test_data.info())

# Check for missing values
print("\nMissing Values in Training Data:")
print(train_data.isnull().sum())

print("\nMissing Values in Test Data:")
print(test_data.isnull().sum())

# Check the distribution of the target variable (class labels)
print("\nClass Distribution in Training Data:")
print(train_data['label'].value_counts(normalize=True))

# Visualize the class distribution
sns.countplot(x='label', data=train_data)
plt.title("Class Distribution in Training Data")
plt.show()

# Check the length of the time series for different objects
train_data['values_length'] = train_data['values'].apply(len)

print("\nStatistics for Time Series Length in Training Data:")
print(train_data['values_length'].describe())

# Plot the distribution of the length of time series
sns.histplot(train_data['values_length'], bins=30)
plt.title("Distribution of Time Series Length in Training Data")
plt.show()

# Visualize some sample time series data
plt.figure(figsize=(10,6))

# Plot a time series from class 0
example_0 = train_data[train_data['label'] == 0]['values'].values[0]
plt.plot(example_0, label='Class 0')

# Plot a time series from class 1
example_1 = train_data[train_data['label'] == 1]['values'].values[0]
plt.plot(example_1, label='Class 1')

plt.legend()
plt.title("Example Time Series from Both Classes")
plt.show()
