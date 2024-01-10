from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from model_base import ModelBase

class CNNModel(ModelBase):
    def build_model(self, input_shape):
        model = Sequential()
        model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model
