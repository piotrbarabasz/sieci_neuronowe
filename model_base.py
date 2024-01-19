from abc import ABC, abstractmethod
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from model_plots import ModelPlots
from sklearn.metrics import balanced_accuracy_score


class ModelBase(ABC):
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)

    @abstractmethod
    def build_model(self, input_shape):
        pass

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, model_name, epochs=100, batch_size=32):
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

        model_file = f'models/results/model_{model_name}.h5'
        self.model.save(model_file)
        print(f"Model saved: {model_file}")

        y_pred = self.model.predict(X_test)
        y_pred = (y_pred > 0.5)

        results = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC Score": roc_auc_score(y_test, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred)
        }

        plotter = ModelPlots({model_name: self.model}, X_test, y_test)
        plotter.plot_roc_curves()
        plotter.plot_precision_recall_curves()
        plotter.plot_confusion_matrices()

        return results