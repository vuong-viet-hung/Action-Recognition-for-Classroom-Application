from pathlib import Path

from tensorflow import keras


DATA_PATH = "../data/UCF101/UCF-101"
KEYPOINTS_DIMS = (120, 33 * 4)
N_LABELS = len([directory.name for directory in Path(DATA_PATH).iterdir()])


class LSTMModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.layers = keras.models.Sequential(
            [
                keras.layers.LSTM(64, return_sequences=True, activation="relu", input_shape=KEYPOINTS_DIMS),
                keras.layers.LSTM(128, return_sequences=True, activation="relu"),
                keras.layers.LSTM(64, return_sequences=False, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(N_LABELS, activation="softmax"),
            ]
        )

    def call(self, inputs):
        return self.layers(inputs)
