from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from model_base import ModelBase

class CNNLSTMModel(ModelBase):
    def build_model(self, input_shape):
        model = Sequential()
        model.add(Conv1D(32, 3, activation='relu', input_shape=input_shape))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        return model
