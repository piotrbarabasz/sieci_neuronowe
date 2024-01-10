from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from model_base import ModelBase

class LSTMModel(ModelBase):
    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
        model.add(LSTM(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        return model
