from data_handler import DataHandler
from models.definition.cnn_lstm_model import CNNLSTMModel
from models.definition.lstm_model import LSTMModel
from models.definition.dense_model import DenseModel
from models.definition.gru_model import GRUModel
from models.definition.cnn_model import CNNModel
from models.definition.mlp_model import MLPModel
import csv

data_handler = DataHandler()
data_handler.load_data()
X_train, y_train, X_test, y_test = data_handler.preprocess_data()

models = {
    "CNN-LSTM": CNNLSTMModel((X_train.shape[1], 1)),
    "LSTM": LSTMModel((X_train.shape[1], 1)),
    "Dense": DenseModel((X_train.shape[1],)),
    "GRU": GRUModel((X_train.shape[1], 1)),
    "CNN": CNNModel((X_train.shape[1], 1)),
    "MLP": MLPModel((X_train.shape[1],))
}

results_file = 'results/training/model_results.csv'

with open(results_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score', 'Balanced Accuracy'])

    for name, model_instance in models.items():
        print(f"Training and evaluating {name} model")
        results = model_instance.train_and_evaluate(
            X_train.reshape(X_train.shape[0], X_train.shape[1], -1),
            y_train,
            X_test.reshape(X_test.shape[0], X_test.shape[1], -1),
            y_test,
            name
        )

        writer.writerow(
            [name] + [results[key] for key in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score', 'Balanced Accuracy']])