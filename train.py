import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, GRU, Flatten
from tensorflow.keras.optimizers import Adam
import csv

train_data = pd.DataFrame()
test_data = pd.DataFrame()

for i in range(20):
    file_path = f'train/data_{i}.csv'
    temp_df = pd.read_csv(file_path)
    train_data = pd.concat([train_data, temp_df], ignore_index=True)

for i in range(20, 25):
    file_path = f'test/data_{i}.csv'
    temp_df = pd.read_csv(file_path)
    test_data = pd.concat([test_data, temp_df], ignore_index=True)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def create_cnn_lstm_model(input_shape):
    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(LSTM(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

def create_dense_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

def create_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(GRU(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def create_mlp_model(input_shape):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    model_file = f'models/model_{model_name}.h5'
    model.save(model_file)
    print(f"Model saved: {model_file}")

    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC Score": roc_auc_score(y_test, y_pred)
    }
    return results


models = {
    "CNN-LSTM": create_cnn_lstm_model((X_train.shape[1], 1)),
    "LSTM": create_lstm_model((X_train.shape[1], 1)),
    "Dense": create_dense_model((X_train.shape[1],)),
    "GRU": create_gru_model((X_train.shape[1], 1)),
    "CNN": create_cnn_model((X_train.shape[1], 1)),
    "MLP": create_mlp_model((X_train.shape[1],))
}
# exit()
results_file = 'results/model_results.csv'

with open(results_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score'])

    for name, model in models.items():
        print(f"Training and evaluating {name} model")
        results = train_and_evaluate(
            model,
            name,
            X_train.reshape(X_train.shape[0], X_train.shape[1], -1),
            y_train,
            X_test.reshape(X_test.shape[0], X_test.shape[1], -1),
            y_test
        )

        writer.writerow(
            [name] + [results[key] for key in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score']])

# model = Sequential()
# model.add(Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
# model.add(Conv1D(64, 3, activation='relu'))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=10, batch_size=32, validation_split=0.2)
# model.save('model.h5')

# y_pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
# y_pred = (y_pred > 0.5)

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_pred)

# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)
# print("ROC AUC Score:", roc_auc)
