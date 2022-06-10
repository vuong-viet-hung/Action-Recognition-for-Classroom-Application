from pathlib import Path

from tensorflow import keras


DATA_PATH = "data/UCF101/UCF-101/"
KEYPOINTS_DIMS = (120, 33 * 4)
N_CLASSES = len([directory.name for directory in Path(DATA_PATH).iterdir()])


class LSTMModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.model = keras.models.Sequential(
            [
                keras.layers.Input(shape=KEYPOINTS_DIMS),
                keras.layers.LSTM(units=64, return_sequences=True, activation="relu"),
                keras.layers.LSTM(units=128, return_sequences=True, activation="relu"),
                keras.layers.LSTM(units=64, return_sequences=False, activation="relu"),
                keras.layers.Dense(units=64, activation="relu"),
                keras.layers.Dense(units=32, activation="relu"),
                keras.layers.Dense(units=N_CLASSES, activation="softmax"),
            ]
        )

    def call(self, inputs):
        return self.model(inputs)
