import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import RepeatedStratifiedKFold
from collections import Counter

class DataHandler:
    def __init__(self):
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.scaler = StandardScaler()
        self.undersampling_pipeline = Pipeline([
            ('smote', SMOTE(sampling_strategy='auto', random_state=42)),
            ('tomek', TomekLinks())
        ])

    def load_data(self):
        for i in range(20):
            file_path = f'dataset/train/data_{i}.csv'
            temp_df = pd.read_csv(file_path)
            self.train_data = pd.concat([self.train_data, temp_df], ignore_index=True)

        for i in range(20, 25):
            file_path = f'dataset/test/data_{i}.csv'
            temp_df = pd.read_csv(file_path)
            self.test_data = pd.concat([self.test_data, temp_df], ignore_index=True)

    def preprocess_data(self):
        X = self.train_data.iloc[:, :-1].values
        y = self.train_data.iloc[:, -1].values
        X = self.scaler.fit_transform(X)

        rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=1410)

        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train_balanced, y_train_balanced = self.undersampling_pipeline.fit_resample(X_train, y_train)

            imbalance_rate_before = self.calculate_imbalance_rate(y_train)
            imbalance_rate_after = self.calculate_imbalance_rate(y_train_balanced)

        print(f"Imbalance Rate Before Resampling: {imbalance_rate_before}")
        print(f"Imbalance Rate After Resampling: {imbalance_rate_after}")

        return X_train_balanced, y_train_balanced, X_test, y_test

    def calculate_imbalance_rate(self, y):
        class_counts = Counter(y)
        majority_class_count = max(class_counts.values())
        minority_class_count = min(class_counts.values())
        imbalance_rate = majority_class_count / minority_class_count
        return imbalance_rate