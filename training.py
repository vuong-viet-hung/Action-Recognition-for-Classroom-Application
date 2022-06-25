from pathlib import Path
from tensorflow import keras

from data_processing import (
    ActionKeypointsGenerator,
    sample_classes,
    sample_data,
    train_valid_test_split,
)
from model import make_model


DATA_PATH = "data/UCF101/UCF-101/"
N_CLASSES = 2
N_SAMPLES_PER_CLASSES = 10
CHECKPOINT_PATH = "checkpoint/"
SAVED_MODEL_PATH = "saved_model/lstm_model.h5"
LOGS_PATH = "logs/"
EPOCHS = 100
BATCH_SIZE = 8


def main() -> None:
    selected_classes_directories = sample_classes(DATA_PATH, N_CLASSES)
    data_paths = sample_data(selected_classes_directories, unique_groups=False)
    string_labels = [data_path.parent.name for data_path in data_paths]

    # Split the data into train and test set
    (
        train_data_paths,
        valid_data_paths,
        test_data_paths,
        train_labels,
        valid_labels,
        test_labels,
    ) = train_valid_test_split(
        data_paths,
        string_labels,
        valid_size=0.1,
        test_size=0.1,
    )

    # Create train and test data generator
    selected_classes = [
        selected_class_directory.name
        for selected_class_directory in selected_classes_directories
    ]

    print(selected_classes)

    train_data_generator = ActionKeypointsGenerator(
        train_data_paths, train_labels, selected_classes, BATCH_SIZE
    )
    valid_data_generator = ActionKeypointsGenerator(
        valid_data_paths, valid_labels, selected_classes, BATCH_SIZE
    )
    test_data_generator = ActionKeypointsGenerator(
        test_data_paths, test_labels, selected_classes, BATCH_SIZE
    )

    # Define callbacks for training
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(Path(CHECKPOINT_PATH))
    )
    early_stopping_callback = keras.callbacks.EarlyStopping(patience=5)
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=str(Path(LOGS_PATH)), histogram_freq=1
    )

    # Train the model
    model = make_model(N_CLASSES)
    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    model.fit(
        train_data_generator,
        callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback],
        validation_data=valid_data_generator,
        epochs=EPOCHS,
    )
    results = model.evaluate(
        test_data_generator,
        callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback],
    )
    print("loss, accuracy = ", results)
    model.save(str(Path(SAVED_MODEL_PATH)))


if __name__ == "__main__":
    main()
