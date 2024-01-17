from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from model_base import ModelBase

class MLPModel(ModelBase):
    def build_model(self, input_shape):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def extract_features(self, X, model_name):
        if not hasattr(self, 'feature_model'):
            self.feature_model = self.build_model(X.shape[1:])
            model_file = f'models/results/_model_{model_name}.h5'
            self.feature_model.load_weights(f'{model_file}', by_name=True)

        features = self.feature_model.predict(X)
        return features