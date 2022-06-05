import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tensorflow import keras

from keypoints_extractor import extract_keypoints


DATA_PATH = "../data/UCF101/UCF-101"


class ActionKeypointsGenerator(keras.utils.Sequence):
    labels = [directory.name for directory in Path(DATA_PATH).iterdir()]
    labels_map = dict(((label, idx) for (idx, label) in enumerate(labels)))

    def __init__(self, video_paths: List[str], batch_size: int, shuffle: bool = True) -> None:
        self.video_paths: List[str] = video_paths
        self.batch_size: int = batch_size
        if shuffle:
            random.shuffle(video_paths)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        data_size = len(self.video_paths)
        n_batches = data_size // self.batch_size
        batch_start = self.batch_size * idx
        batch_end = self.batch_size * (idx + 1) if idx < n_batches else data_size

        keypoints = []
        encoded_labels = []

        for video_path in self.video_paths[batch_start:batch_end]:
            keypoints.append(extract_keypoints(video_path))
            encoded_labels.append(self.labels_map[str(Path(video_path).parent.name)])

        return np.array(keypoints), np.array(encoded_labels)

    def __len__(self) -> int:
        return len(self.video_paths)


def demo() -> None:
    assert Path(DATA_PATH).is_dir()
    video_paths = [str(video_path) for video_path in Path(DATA_PATH).glob("**/*.avi")]
    data_generator = ActionKeypointsGenerator(video_paths, batch_size=8)
    for keypoints, labels in data_generator:
        print(f"{keypoints.shape = }\t{labels.shape = }")


if __name__ == "__main__":
    demo()
