"""
A module for functions that read data information from annotation files
"""

import csv
from typing import Generator
from pathlib import Path


def get_class_indices(
    annotation_file_path: str, selected_classes: set[str] | None = None
) -> dict[int, str]:
    """Return a dict containing class names and respected indices"""

    with open(annotation_file_path, "r", encoding="utf-8") as class_indices_file:
        reader = csv.reader(class_indices_file, delimiter=" ")
        if selected_classes is None:
            return {int(idx): class_name for idx, class_name in reader}
        return {
            int(idx): class_name
            for idx, class_name in reader
            if class_name in selected_classes
        }


def get_test_samples(
    annotation_file_paths: list[str], selected_classes: set[str] | None = None
) -> Generator[str, None, None]:
    """Yield name of video samples used for testing"""

    for file_path in annotation_file_paths:
        with open(file_path, "r", encoding="utf-8") as sample_list:
            for sample in sample_list:
                sample = sample.strip()
                sample_class = str(Path(sample).parent)
                if selected_classes is None or sample_class in selected_classes:
                    yield sample


def get_train_samples(
    annotation_file_paths: list[str], selected_classes: set[str] | None = None
) -> Generator[tuple[str, int], None, None]:
    """Yield name and label of video samples used for training"""

    for file_path in annotation_file_paths:
        with open(file_path, "r", encoding="utf-8") as sample_list:
            reader = csv.reader(sample_list, delimiter=" ")
            for sample, label in reader:
                sample_class = str(Path(sample).parent)
                if selected_classes is None or sample_class in selected_classes:
                    yield sample, int(label)
