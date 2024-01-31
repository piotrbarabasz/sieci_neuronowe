from data_handler import DataHandler
from models.definition.cnn_lstm_model import CNNLSTMModel
from models.definition.lstm_model import LSTMModel
from models.definition.dense_model import DenseModel
from models.definition.gru_model import GRUModel
from models.definition.cnn_model import CNNModel
from models.definition.mlp_model import MLPModel
import csv
import optuna
import mlflow
import tensorflow as tf
import numpy as np

data_handler = DataHandler()
data_handler.load_data()
X_train, y_train, X_test, y_test = data_handler.preprocess_data()


# models = {
#     "CNN-LSTM": CNNLSTMModel((X_train.shape[1], 1)),
#     "LSTM": LSTMModel((X_train.shape[1], 1)),
#     "Dense": DenseModel((X_train.shape[1],)),
#     "GRU": GRUModel((X_train.shape[1], 1)),
#     "CNN": CNNModel((X_train.shape[1], 1)),
#     "MLP": MLPModel((X_train.shape[1],))
# }
models = { "CNN-LSTM": CNNLSTMModel(input_shape=(X_train.shape[1], 1), num_conv_layers=2, num_lstm_units=128, dropout_rate=0.3) }

results_file = 'results/training/model_results.csv'

# Data preparation
# data_optimizer_handler = DataHandler()
# data_optimizer_handler.load_data()
# X_trn, y_trn, X_val, y_val = data_optimizer_handler.split()

def objective(trial):
    with mlflow.start_run():

        # Hyperparameters to tune
        num_conv_layers = trial.suggest_int('num_conv_layers', 1, 3)
        num_lstm_units = trial.suggest_categorical('num_lstm_units', [64, 128, 256])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)


        # Load and preprocess your data
        # X_trn, y_trn = data_optimizer_handler.get_training_data()  # Replace with actual method
        # X_val, y_val = data_optimizer_handler.get_validation_data()  # Replace with actual method

        # Model creation
        model = CNNLSTMModel(input_shape=X_train.shape[1:], num_conv_layers=num_conv_layers, num_lstm_units=num_lstm_units, dropout_rate=dropout_rate)
        optimizer_choice = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

        if optimizer_choice == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Model fitting
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        epochs = trial.suggest_int('epochs', 10, 50)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)

        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(X_test, y_test)

        mlflow.log_params(trial.params)
        mlflow.log_metric('val_loss', val_loss)
        mlflow.log_metric('val_accuracy', val_accuracy)

        # Optionally, log the model
        mlflow.keras.log_model(model, "model")

    # Using validation loss as the metric to minimize
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10) # You can adjust the number of trials

exit()
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