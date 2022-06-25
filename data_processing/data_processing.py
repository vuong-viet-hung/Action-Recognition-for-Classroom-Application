import math
import random
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

from data_processing.keypoints_extractor import extract_keypoints


DATA_PATH = "data/UCF101/UCF-101"


class ActionKeypointsGenerator(keras.utils.Sequence):
    def __init__(
        self,
        data_paths: List[str],
        string_labels: List[str],
        classes: List[str],
        batch_size: int,
    ) -> None:
        self.data_paths: List[str] = data_paths
        self.classes: List[str] = classes
        self.batch_size: int = batch_size
        self.data_size: int = len(self.data_paths)
        self.n_batches: int = math.floor(self.data_size // self.batch_size)

        labels_map = {label: idx for idx, label in enumerate(classes)}
        numeric_labels = [labels_map[string_label] for string_label in string_labels]
        self.encoded_labels: np.ndarray = keras.utils.to_categorical(numeric_labels)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        batch_start = self.batch_size * idx
        batch_end = self.batch_size * (idx + 1) if idx < self.n_batches else self.data_size

        keypoints = []
        labels = []

        for data_path, label in zip(
            self.data_paths[batch_start:batch_end],
            self.encoded_labels[batch_start:batch_end],
        ):
            keypoints.append(extract_keypoints(str(data_path)))
            labels.append(label)

        assert not np.any(np.isnan(np.array(keypoints)))
        assert not np.any(np.isnan(np.array(labels)))

        return np.array(keypoints), np.array(labels)

    def __len__(self) -> int:
        return math.ceil(self.data_size / self.batch_size)


def sample_classes(data_path: str, n_classes: int) -> List[Path]:
    """Yield paths to directories for randomly selected classes"""
    classes_directories = [directory for directory in Path(data_path).iterdir()]
    return random.sample(classes_directories, n_classes)


def sample_data(
    selected_classes_directories: List[Path],
    n_samples_per_classes: int | None = None,
    unique_groups: bool = True,
) -> List[Path]:
    """Yield paths to randomly selected data samples"""
    sample_regex = re.compile(r"v_(\w+)_g(\d+)_c(\d+).avi")
    existing_groups = set()
    selected_samples = []

    for selected_class_directory in selected_classes_directories:
        for selected_sample in selected_class_directory.iterdir():
            matches = sample_regex.search(selected_sample.name)
            # Sample's information contains its action and group
            sample_infos = (matches.group(1), matches.group(2))
            # Append samples from different groups for more data diversity
            if unique_groups and sample_infos in existing_groups:
                continue
            existing_groups.add(sample_infos)
            selected_samples.append(selected_sample)

    if n_samples_per_classes is None:
        return selected_samples

    try:    
        return selected_samples[:n_samples_per_classes]
    except IndexError:
        return selected_samples


def train_valid_test_split(
    data_paths: List[Path],
    labels: List[str],
    valid_size: float,
    test_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the dataset into train, valid and test sets"""
    # Split the dataset into train + valid and test sets
    (
        train_valid_data_paths,
        test_data_paths,
        train_valid_labels,
        test_labels,
    ) = train_test_split(
        data_paths,
        labels,
        test_size=test_size,
        stratify=labels,
    )

    # Split the train + valid set into train and valid sets
    train_data_paths, valid_data_paths, train_labels, valid_labels = train_test_split(
        train_valid_data_paths,
        train_valid_labels,
        test_size=valid_size / (1 - test_size),
        stratify=train_valid_labels,
    )

    return (
        train_data_paths,
        valid_data_paths,
        test_data_paths,
        train_labels,
        valid_labels,
        test_labels,
    )
