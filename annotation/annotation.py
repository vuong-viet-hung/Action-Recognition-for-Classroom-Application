"""
A module for functions that read data information from annotation files
"""

import csv
import random


def get_class_indices(
    annotation_file_path: str, num_selected_class: int | None = None
) -> dict[int, str]:
    """Return a dict containing class indices and respected names"""
    with open(annotation_file_path, "r", encoding="utf-8") as class_indices_file:
        reader = csv.reader(class_indices_file)
        idx_list = [(int(idx), class_name) for idx, class_name in reader]
        if num_selected_class is not None:
            idx_list = random.sample(idx_list, num_selected_class)
        return dict(idx_list)
