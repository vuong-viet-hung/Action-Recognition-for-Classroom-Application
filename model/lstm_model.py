from pathlib import Path

from tensorflow import keras


KEYPOINTS_DIMS = (120, 33 * 4)


class LSTMModel(keras.Model):
    def __init__(self, n_classes):
        super().__init__()
        self.model = keras.models.Sequential(
            [
                keras.layers.Input(shape=KEYPOINTS_DIMS),
                keras.layers.LSTM(units=128, return_sequences=True, activation="relu"),
                keras.layers.LSTM(units=128, return_sequences=True, activation="relu"),
                keras.layers.LSTM(units=256, return_sequences=True, activation="relu"),
                keras.layers.LSTM(units=256, return_sequences=True, activation="relu"),
                keras.layers.LSTM(units=512, return_sequences=True, activation="relu"),
                keras.layers.LSTM(units=512, return_sequences=True, activation="relu"),
                keras.layers.LSTM(units=256, return_sequences=True, activation="relu"),
                keras.layers.LSTM(units=256, return_sequences=False, activation="relu"),
                keras.layers.Dense(units=256, activation="relu"),
                keras.layers.Dense(units=256, activation="relu"),
                keras.layers.Dense(units=128, activation="relu"),
                keras.layers.Dense(units=128, activation="relu"),
                keras.layers.Dense(units=n_classes, activation="softmax"),
            ]
        )

    def call(self, inputs):
        return self.model(inputs)
