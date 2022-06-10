from tensorflow import keras


KEYPOINTS_DIMS = (128, 33 * 4)


def make_model(n_classes):
    return keras.models.Sequential(
        [
            keras.layers.Input(shape=KEYPOINTS_DIMS),
            keras.layers.LSTM(units=128, return_sequences=True, activation="relu"),
            keras.layers.LSTM(units=256, return_sequences=True, activation="relu"),
            keras.layers.LSTM(units=512, return_sequences=True, activation="relu"),
            keras.layers.LSTM(units=256, return_sequences=True, activation="relu"),
            keras.layers.LSTM(units=128, return_sequences=False, activation="relu"),
            keras.layers.Dense(units=128, activation="relu"),
            keras.layers.Dense(units=64, activation="relu"),
            keras.layers.Dense(units=32, activation="relu"),
            keras.layers.Dense(units=n_classes, activation="softmax"),
        ]
    )
