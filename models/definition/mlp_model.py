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
