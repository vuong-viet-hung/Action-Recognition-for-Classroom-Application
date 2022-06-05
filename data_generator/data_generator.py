import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tensorflow import keras

from keypoints_extractor import extract_keypoints


DATA_PATH = "../data/UCF101/UCF-101"


def encode_label(label: str) -> int:
    labels = [directory.name for directory in Path(DATA_PATH).iterdir()]
    labels_map = dict(((label, idx) for (idx, label) in enumerate(labels)))
    return labels_map[label]


class ActionKeypointsGenerator(keras.utils.Sequence):
    def __init__(self, data_paths: List[str], labels: List[int], batch_size: int) -> None:
        self.data_paths: List[str] = data_paths
        self.labels: List[int] = labels
        self.batch_size: int = batch_size

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        data_size = len(self.data_paths)
        n_batches = data_size // self.batch_size
        batch_start = self.batch_size * idx
        batch_end = self.batch_size * (idx + 1) if idx < n_batches else data_size

        keypoints = []
        labels = []

        for data_path, label in zip(self.data_paths[batch_start:batch_end], self.labels[batch_start:batch_end]):
            keypoints.append(extract_keypoints(data_path))
            labels.append(label)

        return np.array(keypoints), np.array(labels)

    def __len__(self) -> int:
        return len(self.data_paths)


def demo() -> None:
    assert Path(DATA_PATH).is_dir()
    data_paths = [str(data_path) for data_path in Path(DATA_PATH).glob("**/*.avi")]
    labels = [encode_label(Path(data_path).parent.name) for data_path in data_paths]
    data_generator = ActionKeypointsGenerator(data_paths, labels, batch_size=8)
    for keypoints, labels in data_generator:
        print(f"{keypoints.shape = }\t{labels.shape = }")


if __name__ == "__main__":
    demo()
