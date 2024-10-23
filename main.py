import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load the data (assuming the train.parquet is already loaded as 'train_data')
train_data = pd.read_parquet('train.parquet')

# Initialize the feature list and the target (label) list
features = []
labels = []

# Loop over the rows in train_data
for index, row in train_data.iterrows():
    values = row['values']  # The time series data
    label = row['label']    # The corresponding label

    # Basic statistical features extracted from the time series
    feature_dict = {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values),
        'sum': np.sum(values),
        'range': np.max(values) - np.min(values),  # Range (max - min)
        'var': np.var(values),                     # Variance
        'skewness': pd.Series(values).skew(),      # Skewness
        'kurtosis': pd.Series(values).kurtosis(),  # Kurtosis
    }

    # Append the feature dictionary to the list of features
    features.append(feature_dict)

    # Append the label to the list of labels
    labels.append(label)

# Convert the list of features into a DataFrame
X = pd.DataFrame(features)

# Convert the labels into a Series
y = pd.Series(labels, name='label')

# Check the shape of the extracted features and labels
# print(f"X shape: {X.shape}, y shape: {y.shape}")

# Display the first few rows of the extracted feature DataFrame
# print(X.head())


# Split the data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Initialize XGBoost classifier
# xgb_model = xgb.XGBClassifier(
#     objective='binary:logistic', 
#     scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),  # Handle class imbalance
#     eval_metric='auc',
#     use_label_encoder=False
# )

# # Train the model
# xgb_model.fit(X_train, y_train)

# # Predict probabilities for the validation set
# y_pred_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]

# # Evaluate the model using ROC AUC
# roc_auc_xgb = roc_auc_score(y_val, y_pred_proba_xgb)
# print(f"XGBoost ROC AUC: {roc_auc_xgb:.4f}")


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_auc_score

# # Initialize Random Forest classifier
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# # Train the model
# rf_model.fit(X_train, y_train)

# # Predict probabilities for the validation set
# y_pred_proba_rf = rf_model.predict_proba(X_val)[:, 1]

# # Evaluate the model using ROC AUC
# roc_auc_rf = roc_auc_score(y_val, y_pred_proba_rf)
# print(f"Random Forest ROC AUC: {roc_auc_rf:.4f}")


### LSTM ###
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score

# Check for NaNs in each sequence within train_data['values']
import numpy as np

def check_for_nan_in_sequences(data):
    nan_count = 0
    for seq in data:
        if isinstance(seq, (list, np.ndarray)):
            nan_count += np.isnan(seq).sum()  # Check each sequence for NaNs
            if len(seq) == nan_count:
                print('All NaN')
        else:
            print(f"Unexpected data type: {type(seq)}")  # For debugging non-sequence entries
    return nan_count

nan_count = check_for_nan_in_sequences(train_data['values'])
print(f"Total NaNs found: {nan_count}")


# 0.0
train_data['values'] = train_data['values'].apply(lambda seq: np.nan_to_num(seq, nan=0.0))
# mean
# train_data['values'] = train_data['values'].apply(lambda seq: [np.nanmean(seq)] if np.isnan(seq).any() else seq)
# median
# train_data['values'] = train_data['values'].apply(lambda seq: np.nan_to_num(seq, nan=np.nanmedian(seq)))
# -1
# train_data['values'] = train_data['values'].apply(lambda seq: np.nan_to_num(seq, nan=-1))

nan_count = check_for_nan_in_sequences(train_data['values'])
print(f"Total NaNs found: {nan_count}")

# Prepare the input sequences (time series data)
X_sequences = pad_sequences(train_data['values'], dtype='float32', padding='post', maxlen=97)
y_sequences = train_data['label'].values

# Split the data into training and validation sets
X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences)
# mask = ~np.isnan(X_train_seq).any(axis=1)
# X_train_clean = X_train_seq[mask]
# y_train_clean = y_train_seq[mask]  # Ensure corresponding labels are also filtered

# # Do the same for the validation set (if necessary)
# X_val_clean = X_val_seq[~np.isnan(X_val_seq).any(axis=1)]
# y_val_clean = y_val_seq[~np.isnan(X_val_seq).any(axis=1)]


# Define the LSTM model
lstm_model = Sequential([
    Masking(mask_value=0.0, input_shape=(97, 1)),  # Masking padded values
    LSTM(64, return_sequences=False),
    Dense(1, activation='sigmoid')
])

# Compile the model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# Train the model
# lstm_model.fit(X_train_clean, y_train_clean, validation_data=(X_val_clean, y_val_clean), epochs=10, batch_size=64)
lstm_model.fit(X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq), epochs=10, batch_size=64)

# Predict probabilities for the validation set
# y_pred_proba_lstm = lstm_model.predict(X_val_clean).flatten()
y_pred_proba_lstm = lstm_model.predict(X_val_seq).flatten()

# Evaluate the model using ROC AUC
# roc_auc_lstm = roc_auc_score(y_val_clean, y_pred_proba_lstm)
roc_auc_lstm = roc_auc_score(y_val_seq, y_pred_proba_lstm)
print(f"LSTM ROC AUC: {roc_auc_lstm:.4f}")


### CNN ###
# from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D

# # Define the 1D CNN model
# cnn_model = Sequential([
#     Masking(mask_value=0.0, input_shape=(97, 1)),  # Masking padded values
#     Conv1D(filters=64, kernel_size=3, activation='relu'),
#     GlobalAveragePooling1D(),
#     Dense(1, activation='sigmoid')
# ])

# # Compile the model
# cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# # Train the model
# cnn_model.fit(X_train_clean, y_train_clean, validation_data=(X_val_clean, y_val_clean), epochs=10, batch_size=64)

# # Predict probabilities for the validation set
# y_pred_proba_cnn = cnn_model.predict(X_val_clean).flatten()

# # Evaluate the model using ROC AUC
# roc_auc_cnn = roc_auc_score(y_val_clean, y_pred_proba_cnn)
# print(f"1D CNN ROC AUC: {roc_auc_cnn:.4f}")
