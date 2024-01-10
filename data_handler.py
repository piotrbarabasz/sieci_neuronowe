import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataHandler:
    def __init__(self):
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.scaler = StandardScaler()

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
        X_train = self.train_data.iloc[:, :-1].values
        y_train = self.train_data.iloc[:, -1].values
        X_test = self.test_data.iloc[:, :-1].values
        y_test = self.test_data.iloc[:, -1].values

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, y_train, X_test, y_test
