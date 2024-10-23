import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the test data
test_data = pd.read_parquet('test.parquet')

# Replace NaNs in the test set
test_data['values'] = test_data['values'].apply(lambda seq: np.nan_to_num(seq, nan=0.0))

# Prepare input sequences for the test data
X_test_seq = pad_sequences(test_data['values'], dtype='float32', padding='post', maxlen=97)

# Load the trained model
lstm_model = load_model('lstm_model.h5')
print("Loaded trained model.")

# Generate predictions (probabilities) for the test set
y_pred_proba_test = lstm_model.predict(X_test_seq).flatten()

# Create a submission dataframe
submission = pd.DataFrame({
    'id': test_data['id'],         # Use the 'id' from the test data
    'score': y_pred_proba_test     # Use the predicted probabilities for class 1
})

# Save the predictions to a CSV file
submission_file = 'submission.csv'
submission.to_csv(submission_file, index=False)
print(f"Submission file created: {submission_file}")
