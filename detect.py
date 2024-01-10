import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

def detect_anomalies(model, data, threshold=0.5):
    predictions = model.predict(data)
    anomalies = predictions > threshold
    return anomalies.flatten()

cnn_lstm_model = load_model('model_CNN-LSTM.h5')
lstm_model = load_model('model_LSTM.h5')
dense_model = load_model('model_Dense.h5')

new_data_df = pd.read_csv('test/data_20.csv')
X_new = new_data_df.drop('anomaly', axis=1).values

scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

X_new_scaled_cnn_lstm = X_new_scaled.reshape(X_new_scaled.shape[0], X_new_scaled.shape[1], 1)
X_new_scaled_lstm = X_new_scaled.reshape(X_new_scaled.shape[0], X_new_scaled.shape[1], 1)

anomalies_cnn_lstm = detect_anomalies(cnn_lstm_model, X_new_scaled_cnn_lstm)
anomalies_lstm = detect_anomalies(lstm_model, X_new_scaled_lstm)
anomalies_dense = detect_anomalies(dense_model, X_new_scaled)

results_df = pd.DataFrame({
    'CNN-LSTM': anomalies_cnn_lstm,
    'LSTM': anomalies_lstm,
    'Dense': anomalies_dense
})

# results_df.to_csv('anomalies_detected.csv', index=False)
print("Anomaly detection complete. Results saved to anomalies_detected.csv.")


anomalies_indices = results_df.index[results_df['CNN-LSTM'] == 1].tolist()


plt.figure(figsize=(14, 7))
for col in new_data_df.columns[:-1]:  # Exclude the 'anomaly' column
    plt.plot(new_data_df[col], label=col)
plt.scatter(anomalies_indices, new_data_df.loc[anomalies_indices, new_data_df.columns[0]], color='r', label='Anomalies')
plt.title('Anomalies Detected in Data')
plt.xlabel('Data Point Index')
plt.ylabel('Sensor Readings')
plt.legend()
plt.grid(True)