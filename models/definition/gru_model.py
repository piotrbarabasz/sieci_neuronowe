from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from model_base import ModelBase

class GRUModel(ModelBase):
    def build_model(self, input_shape):
        model = Sequential()
        model.add(GRU(64, return_sequences=True, activation='relu', input_shape=input_shape))
        model.add(GRU(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        return model
