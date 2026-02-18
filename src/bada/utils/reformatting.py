import numpy as np

from bada.processing.feature_extraction import DSFCurveFeatures
from bada.utils.utils import get_column_labels, get_row_labels

"""
This needs some refactoring, plenty of code duplication.
"""


def convert_distances_to_plate_format(
    distances: dict[str, tuple[float, str]], plate_size: int
) -> tuple[np.ndarray, list[str], list[str]]:
    """Convert DTW distances dictionary to plate format."""
    rows = get_row_labels(plate_size)
    cols = get_column_labels(plate_size, as_str=True)

    plate_data = np.full((len(rows), len(cols)), np.nan)

    for well, (distance, _) in distances.items():
        row_idx = rows.index(well[0])
        col_idx = int(well[1:]) - 1
        plate_data[row_idx, col_idx] = distance

    return plate_data, [str(c) for c in cols], rows


def convert_features_to_plate_format(
    feature_data: dict[str, DSFCurveFeatures],
    plate_size: int,
    feature_name: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Convert DTW distances dictionary to plate format."""
    rows = get_row_labels(plate_size)
    cols = get_column_labels(plate_size, as_str=True)

    plate_data = np.full((len(rows), len(cols)), np.nan)

    for well, features in feature_data.items():
        row_idx = rows.index(well[0])
        col_idx = int(well[1:]) - 1
        plate_data[row_idx, col_idx] = features[feature_name]

    return plate_data, [str(c) for c in cols], rows
