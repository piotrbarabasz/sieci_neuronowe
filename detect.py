import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

def detect_anomalies(model, data, threshold=0.5):
    predictions = model.predict(data)
    anomalies = predictions > threshold
    return anomalies.flatten()

new_data_df = pd.read_csv('dataset/test/data_20.csv')
X_new = new_data_df.drop('anomaly', axis=1).values

scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)
X_new_scaled_reshaped = X_new_scaled.reshape(X_new_scaled.shape[0], X_new_scaled.shape[1], 1)

model_names = ['CNN-LSTM', 'LSTM', 'Dense', 'GRU', 'CNN', 'MLP']
models = {name: load_model(f'models/results/model_{name}.h5') for name in model_names}
anomaly_results = {}

num_models = len(model_names)
cols = 2
rows = num_models // cols + (num_models % cols > 0)
fig, axes = plt.subplots(rows, cols, figsize=(14 * cols, 7 * rows))

for idx, model_name in enumerate(model_names):
    if model_name in ['CNN-LSTM', 'LSTM', 'GRU', 'CNN']:
        anomalies = detect_anomalies(models[model_name], X_new_scaled_reshaped)
    else:  # For Dense and MLP
        anomalies = detect_anomalies(models[model_name], X_new_scaled)

    anomalies_indices = [i for i, x in enumerate(anomalies) if x]
    anomaly_results[model_name] = anomalies

    ax = axes[idx // cols, idx % cols] if num_models > 1 else axes[idx]
    for col in new_data_df.columns[:-1]:
        ax.plot(new_data_df[col], label=col)
    ax.scatter(anomalies_indices, new_data_df.loc[anomalies_indices, new_data_df.columns[0]], color='r', label='Anomalies')
    ax.set_title(f'Anomalies Detected in Data using {model_name}')
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel('Sensor Readings')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
combined_output_file = 'results/detection/all_models_anomalies.png'
plt.savefig(combined_output_file, bbox_inches='tight')
plt.close()

anomalies_df = pd.DataFrame(anomaly_results)
anomalies_csv_file = 'results/detection/consolidated_anomalies.csv'
anomalies_df.to_csv(anomalies_csv_file, index=False)

print("Anomaly detection complete. Results saved to respective files.")
