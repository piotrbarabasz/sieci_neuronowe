from data_handler import DataHandler
from models.definition.cnn_lstm_model import CNNLSTMModel
from models.definition.lstm_model import LSTMModel
from models.definition.dense_model import DenseModel
from models.definition.gru_model import GRUModel
from models.definition.cnn_model import CNNModel
from models.definition.mlp_model import MLPModel
import csv
from sklearn.svm import OneClassSVM
from model_plots import ModelPlots
import joblib


data_handler = DataHandler()
data_handler.load_data()
X_train, y_train, X_test, y_test = data_handler.preprocess_data()

models = {
    # "CNN-LSTM": CNNLSTMModel((X_train.shape[1], 1)),
    # "LSTM": LSTMModel((X_train.shape[1], 1)),
    # "Dense": DenseModel((X_train.shape[1],)),
    # "GRU": GRUModel((X_train.shape[1], 1)),
    # "CNN": CNNModel((X_train.shape[1], 1)),
    "MLP": MLPModel((X_train.shape[1],))
}

results_file = 'results/training/model_results.csv'

with open(results_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score'])

    for name, model_instance in models.items():
        print(f"Training and evaluating {name} model")
        results = model_instance.train_and_evaluate(
            X_train.reshape(X_train.shape[0], X_train.shape[1], -1),
            y_train,
            X_test.reshape(X_test.shape[0], X_test.shape[1], -1),
            y_test,
            name
        )
        X_train_features = model_instance.extract_features(X_train, name)
        X_test_features = model_instance.extract_features(X_test, name)

        print(f"SVM training {name} model")
        oc_svm = OneClassSVM(gamma='auto', nu=0.1)
        oc_svm.fit(X_train_features)

        svm_model_filename = f"models/results/model_{name}_one_class_svm.joblib"
        joblib.dump(oc_svm, svm_model_filename)
        print(f"Saved SVM model to {svm_model_filename}")

        print(f"SVM predicting {name} model")
        y_pred_train = oc_svm.predict(X_train_features)
        y_pred_test = oc_svm.predict(X_test_features)

        print(f"SVM {name} model finished")
        plotter = ModelPlots({name: oc_svm}, X_test_features, y_pred_test)
        plotter.plot_roc_curves()
        plotter.plot_precision_recall_curves()
        plotter.plot_confusion_matrices()
        print(f"Plots created")

        writer.writerow(
            [name] + [results[key] for key in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score']])


