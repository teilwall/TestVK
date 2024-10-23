import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the training data
train_data = pd.read_parquet('train.parquet')

# Replace NaNs in the training set
train_data['values'] = train_data['values'].apply(lambda seq: np.nan_to_num(seq, nan=0.0))

# Prepare input sequences for training
X_sequences = pad_sequences(train_data['values'], dtype='float32', padding='post', maxlen=97)
y_sequences = train_data['label'].values

# Flatten the sequences to 2D for XGBoost (time steps collapsed into features)
X_sequences_flat = X_sequences.reshape(X_sequences.shape[0], -1)

# Split the data into training and validation sets
X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
    X_sequences_flat, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences)

# Initialize the XGBoost model
xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')

# Train the model
xgb_model.fit(X_train_seq, y_train_seq)

# Predict probabilities for the validation set
y_pred_proba_xgb = xgb_model.predict_proba(X_val_seq)[:, 1]

# Convert probabilities to binary predictions with a threshold of 0.5
y_pred_xgb = (y_pred_proba_xgb >= 0.5).astype(int)

# Calculate ROC AUC score
roc_auc_xgb = roc_auc_score(y_val_seq, y_pred_proba_xgb)

# Calculate additional metrics: accuracy, precision, recall, F1 score
accuracy_xgb = accuracy_score(y_val_seq, y_pred_xgb)
precision_xgb = precision_score(y_val_seq, y_pred_xgb)
recall_xgb = recall_score(y_val_seq, y_pred_xgb)
f1_xgb = f1_score(y_val_seq, y_pred_xgb)

# Print the metrics
print(f"XGBoost Validation ROC AUC: {roc_auc_xgb:.4f}")
print(f"XGBoost Validation Accuracy: {accuracy_xgb:.4f}")
print(f"XGBoost Validation Precision: {precision_xgb:.4f}")
print(f"XGBoost Validation Recall: {recall_xgb:.4f}")
print(f"XGBoost Validation F1 Score: {f1_xgb:.4f}")

# Save the trained XGBoost model
xgb_model.save_model('xgb_model.json')
print("XGBoost model saved as xgb_model.json")
