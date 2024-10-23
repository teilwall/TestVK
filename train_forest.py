import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# Load the training data
train_data = pd.read_parquet('train.parquet')

# Replace NaNs in the training set
train_data['values'] = train_data['values'].apply(lambda seq: np.nan_to_num(seq, nan=0.0))

# Prepare input sequences for training
X_sequences = pad_sequences(train_data['values'], dtype='float32', padding='post', maxlen=97)
y_sequences = train_data['label'].values

# Flatten the sequences to 2D for Random Forest (time steps collapsed into features)
X_sequences_flat = X_sequences.reshape(X_sequences.shape[0], -1)

# Split the data into training and validation sets
X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
    X_sequences_flat, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the model
rf_model.fit(X_train_seq, y_train_seq)

# Predict probabilities for the validation set
y_pred_proba_rf = rf_model.predict_proba(X_val_seq)[:, 1]

# Convert probabilities to binary predictions with a threshold of 0.5
y_pred_rf = (y_pred_proba_rf >= 0.5).astype(int)

# Calculate ROC AUC score
roc_auc_rf = roc_auc_score(y_val_seq, y_pred_proba_rf)

# Calculate additional metrics: accuracy, precision, recall, F1 score
accuracy_rf = accuracy_score(y_val_seq, y_pred_rf)
precision_rf = precision_score(y_val_seq, y_pred_rf)
recall_rf = recall_score(y_val_seq, y_pred_rf)
f1_rf = f1_score(y_val_seq, y_pred_rf)

# Print the metrics
print(f"Random Forest Validation ROC AUC: {roc_auc_rf:.4f}")
print(f"Random Forest Validation Accuracy: {accuracy_rf:.4f}")
print(f"Random Forest Validation Precision: {precision_rf:.4f}")
print(f"Random Forest Validation Recall: {recall_rf:.4f}")
print(f"Random Forest Validation F1 Score: {f1_rf:.4f}")

# Save the trained Random Forest model
joblib.dump(rf_model, 'rf_model.joblib')
print("Random Forest model saved as rf_model.joblib")
