import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import numpy as np

class ModelPlots:
    def __init__(self, models, X_test, y_test):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test

    def plot_roc_curves(self):
        plt.figure(figsize=(10, 8))
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test).ravel()
            fpr, tpr, _ = roc_curve(self.y_test, y_pred)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='darkgrey', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for All Models')
        plt.legend(loc="lower right")
        plt.savefig('results/training/all_models_roc.png', bbox_inches='tight')
        plt.close()

    def plot_precision_recall_curves(self):
        plt.figure(figsize=(10, 8))
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test).ravel()
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f'{model_name} (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for All Models')
        plt.legend(loc="lower left")
        plt.savefig('results/training/all_models_precision_recall.png', bbox_inches='tight')
        plt.close()

    def plot_confusion_matrices(self):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        fig.suptitle('Confusion Matrices for All Models')
        for idx, (model_name, model) in enumerate(self.models.items()):
            y_pred = model.predict(self.X_test).ravel()
            y_pred_binary = np.where(y_pred > 0.5, 1, 0)
            cm = confusion_matrix(self.y_test, y_pred_binary)
            ax = axes[idx // 3, idx % 3]
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            ax.set_title(model_name)
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig('results/training/all_models_confusion_matrices.png', bbox_inches='tight')
        plt.close()
