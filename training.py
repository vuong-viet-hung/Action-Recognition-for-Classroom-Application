from pathlib import Path

from sklearn.model_selection import train_test_split

from data_generator import ActionKeypointsGenerator, encode_label
from model import LSTMModel


DATA_PATH = "data/UCF101/UCF-101/"
BATCH_SIZE = 32


def main() -> None:
    assert Path(DATA_PATH).is_dir()

    data_paths = [str(data_path) for data_path in Path(DATA_PATH).glob("**/*.avi")]
    labels = [encode_label(Path(data_path).parent.name) for data_path in data_paths]

    # Split the data into train and test set
    train_data_paths, train_labels, test_data_paths, test_labels = train_test_split(data_paths, labels, stratify=labels)

    # Create train and test data generator
    train_data_generator = ActionKeypointsGenerator(train_data_paths, labels, batch_size=BATCH_SIZE)
    test_data_generator = ActionKeypointsGenerator(test_data_paths, labels, batch_size=BATCH_SIZE)

    model = LSTMModel()
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    model.fit(train_data_generator, test_data_generator)


if __name__ == "__main__":
    main()
