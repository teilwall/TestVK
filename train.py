import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



# Load the data (assuming the train.parquet is already loaded as 'train_data')
train_data = pd.read_parquet('train.parquet')

# replace Nan with 0.0
train_data['values'] = train_data['values'].apply(lambda seq: np.nan_to_num(seq, nan=0.0))

# Prepare the input sequences (time series data)
X_sequences = pad_sequences(train_data['values'], dtype='float32', padding='post', maxlen=97)
y_sequences = train_data['label'].values

# Split the data into training and validation sets
X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences)


# Define the LSTM model
lstm_model = Sequential([
    Masking(mask_value=0.0, input_shape=(97, 1)),  # Masking padded values
    LSTM(64, return_sequences=False),
    Dense(1, activation='sigmoid')
])

# Compile the model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# Train the model
lstm_model.fit(X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq), epochs=10, batch_size=64)

# Predict probabilities for the validation set
y_pred_proba_lstm = lstm_model.predict(X_val_seq).flatten()
y_pred_val = (y_pred_proba_lstm >= 0.5).astype(int)

# Evaluate the model
accuracy_val = accuracy_score(y_val_seq, y_pred_val)
precision_val = precision_score(y_val_seq, y_pred_val)
recall_val = recall_score(y_val_seq, y_pred_val)
f1_val = f1_score(y_val_seq, y_pred_val)
roc_auc_lstm = roc_auc_score(y_val_seq, y_pred_proba_lstm)

print(f"Validation Accuracy: {accuracy_val:.4f}")
print(f"Validation Precision: {precision_val:.4f}")
print(f"Validation Recall: {recall_val:.4f}")
print(f"Validation F1 Score: {f1_val:.4f}")
print(f"LSTM ROC AUC: {roc_auc_lstm:.4f}")

# Save the trained model
lstm_model.save('lstm_model.h5')
print("Model saved as lstm_model.h5")