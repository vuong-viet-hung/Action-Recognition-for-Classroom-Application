from pathlib import Path

import keras.callbacks
from sklearn.model_selection import train_test_split

from data_generator import ActionKeypointsGenerator
from model import LSTMModel


DATA_PATH = "data/UCF101/UCF-101/"
CHECKPOINT_PATH = "saved_model/lstm_model.hdf5"
LOGS_PATH = "logs/"
BATCH_SIZE = 8


def main() -> None:
    assert Path(DATA_PATH).is_dir()

    data_paths = [str(data_path) for data_path in Path(DATA_PATH).glob("**/*.avi")]
    labels = [Path(data_path).parent.name for data_path in data_paths]

    # Split the data into train and test set
    train_data_paths, test_data_paths, train_labels, test_labels = train_test_split(data_paths, labels, stratify=labels)

    # Create train and test data generator
    train_data_generator = ActionKeypointsGenerator(train_data_paths, train_labels, batch_size=BATCH_SIZE)
    test_data_generator = ActionKeypointsGenerator(test_data_paths, test_labels, batch_size=BATCH_SIZE)

    # Define callbacks for training
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=str(Path(CHECKPOINT_PATH)), save_best_only=True)
    early_stopping_callback = keras.callbacks.EarlyStopping(patience=5)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(Path(LOGS_PATH)), histogram_freq=1)

    # Train the model
    model = LSTMModel()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    model.fit(
        train_data_generator,
        callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback],
        validation_data=test_data_generator,
    )


if __name__ == "__main__":
    main()
