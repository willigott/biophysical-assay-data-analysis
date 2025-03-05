from dtaidistance import dtw
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


def get_normalized_signal(signal: np.ndarray | pd.Series) -> np.ndarray:
    """Normalize signal to [0,1] range"""

    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))


def get_spline(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    smoothing: float = 0.01,
    n_points: int = 1000,
) -> tuple[UnivariateSpline, np.ndarray, np.ndarray]:
    """Fit spline to temperature and fluorescence data"""
    spline = UnivariateSpline(x, y, s=smoothing)
    x_spline = np.linspace(min(x), max(x), n_points)
    y_spline = np.asarray(spline(x_spline))

    return (spline, x_spline, y_spline)


def get_spline_derivative(spline: UnivariateSpline, x_spline: np.ndarray) -> np.ndarray:
    """Get derivative of spline"""

    return np.asarray(spline.derivative()(x_spline))


def _get_dtw_distance_unnormalized(
    signal1: np.ndarray | pd.Series, signal2: np.ndarray | pd.Series
) -> float:
    """Calculate DTW distance between two signals"""

    return float(dtw.distance(signal1, signal2))


def _get_dtw_distance_normalized(
    signal1: np.ndarray | pd.Series, signal2: np.ndarray | pd.Series
) -> float:
    """Calculate DTW distance between two normalized signals"""
    norm_signal1 = get_normalized_signal(signal1)
    norm_signal2 = get_normalized_signal(signal2)

    return float(dtw.distance(norm_signal1, norm_signal2))


def get_dtw_distance(
    signal1: np.ndarray | pd.Series, signal2: np.ndarray | pd.Series, normalized: bool = True
) -> float:
    """Calculate DTW distance between two signals with optional normalization
    Args:
        signal1: First signal
        signal2: Second signal
        normalized: If True, normalize signals before calculating distance
    Returns:
        DTW distance between signals
    """
    if normalized:
        return _get_dtw_distance_normalized(signal1, signal2)
    return _get_dtw_distance_unnormalized(signal1, signal2)


def get_dtw_distances_from_reference(
    dsf: pd.DataFrame, reference_well: str, normalized: bool = True
) -> dict[str, tuple[float, str]]:
    """
    Calculate DTW distances from a reference well to all other wells
    Args:
        dsf: DataFrame with columns ['well_position', 'temperature', 'fluorescence']
        reference_well: Well position to use as reference
        normalized: If True, normalize signals before calculating distance
    Returns:
        dictionary mapping well positions to (distance, reference_well)
    """
    wells = dsf["well_position"].unique()
    reference_signal = dsf.loc[dsf["well_position"] == reference_well, "fluorescence"].values

    distances = {}
    for well in wells:
        if well == reference_well:
            distances[well] = (0.0, reference_well)
        else:
            signal = dsf.loc[dsf["well_position"] == well, "fluorescence"].values
            distance = get_dtw_distance(reference_signal, signal, normalized=normalized)
            distances[well] = (distance, reference_well)

    return distances
