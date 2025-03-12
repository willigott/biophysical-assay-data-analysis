from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from bada.processing.preprocessing import get_spline, get_spline_derivative
from bada.utils.validation import validate_temperature_range


def get_min_max_values(
    x: np.ndarray | pd.Series, y: np.ndarray | pd.Series, **kwargs
) -> tuple[float, float, float, float]:
    """Get minimum and maximum values from spline fit

    Returns:
        tuple[float, float, float, float]: (min_value_y, max_value_y, x_at_min, x_at_max)
    """
    _, x_spline, y_spline = get_spline(x, y, **kwargs)

    min_idx = np.argmin(y_spline)
    max_idx = np.argmax(y_spline)

    y_min = y_spline[min_idx]
    y_max = y_spline[max_idx]

    x_at_min_y = x_spline[min_idx]
    x_at_max_y = x_spline[max_idx]

    return (y_min, y_max, x_at_min_y, x_at_max_y)


def _get_max_derivative(spline: BSpline, x_spline: np.ndarray) -> tuple[float, float]:
    """Get maximum derivative from spline fit"""
    y_spline_derivative = get_spline_derivative(spline, x_spline)
    max_derivative_idx = np.argmax(y_spline_derivative)
    x_at_max_derivative = x_spline[max_derivative_idx]
    max_derivative_value = y_spline_derivative[max_derivative_idx]

    return (max_derivative_value, x_at_max_derivative)


def get_tm(
    temperature: np.ndarray | pd.Series, fluorescence: np.ndarray | pd.Series, **kwargs
) -> tuple[float, float]:
    """Get melting temperature (Tm) from signal"""
    spline, x_spline, _ = get_spline(temperature, fluorescence, **kwargs)
    max_derivative_value, tm = _get_max_derivative(spline, x_spline)

    return (tm, max_derivative_value)


def get_dsf_curve_features(
    data: pd.DataFrame,
    min_temp: Optional[float] = None,
    max_temp: Optional[float] = None,
    smoothing: float = 0.01,
    avg_control_tm: Optional[float] = None,
) -> dict[str, float | pd.DataFrame | np.ndarray]:
    """
    Analyze the data for a single well.

    Args:
        data (pd.DataFrame): Dataset pre-filtered for a single well
        min_temp (float, optional): Minimum temperature for analysis range
        max_temp (float, optional): Maximum temperature for analysis range
        smoothing (float, default=0.01): Smoothing factor for spline fitting
        avg_control_tm (float, optional): Average Tm of control wells

    Returns:
        dict: Dictionary containing analysis results

    Raises:
        ValueError: If data contains more than one unique well position
    """

    if n_unique := data["well_position"].nunique() != 1:
        raise ValueError(f"Data must contain exactly one well, but found {n_unique} wells")

    min_fluorescence, max_fluorescence, temp_at_min, temp_at_max = get_min_max_values(
        np.asarray(data["temperature"]), np.asarray(data["fluorescence"]), smoothing=smoothing
    )

    filtered_data = data.copy().reset_index(drop=True)
    if min_temp is None:
        min_temp = float(data["temperature"].min())
    if max_temp is None:
        max_temp = float(data["temperature"].max())

    validate_temperature_range(min_temp, max_temp)

    filtered_data = data[(data["temperature"] >= min_temp) & (data["temperature"] <= max_temp)]

    spline, x_spline, y_spline = get_spline(
        np.asarray(filtered_data["temperature"]),
        np.asarray(filtered_data["fluorescence"]),
        smoothing=smoothing,
    )

    y_spline_derivative = np.asarray(spline.derivative()(x_spline))
    temp_at_max_derivative, max_derivative_value = get_tm(
        np.asarray(filtered_data["temperature"]),
        np.asarray(filtered_data["fluorescence"]),
        smoothing=smoothing,
    )

    if avg_control_tm is not None:
        delta_tm = temp_at_max_derivative - avg_control_tm
    else:
        delta_tm = np.nan

    return {
        "full_well_data": data,
        "x_spline": x_spline,
        "y_spline": y_spline,
        "y_spline_derivative": y_spline_derivative,
        "min_fluorescence": min_fluorescence,
        "max_fluorescence": max_fluorescence,
        "fluorescence_range": max_fluorescence - min_fluorescence,
        "temp_at_min": temp_at_min,
        "temp_at_max": temp_at_max,
        "tm": temp_at_max_derivative,
        "max_derivative_value": max_derivative_value,
        "delta_tm": delta_tm,
        "smoothing": smoothing,
        "min_temp": min_temp,
        "max_temp": max_temp,
    }


def get_dsf_curve_features_multiple_wells(
    data: pd.DataFrame,
    selected_wells: Optional[list[str]] = None,
    min_temp: Optional[float] = None,
    max_temp: Optional[float] = None,
    smoothing: float = 0.01,
    avg_control_tm: Optional[float] = None,
) -> Dict[str, Dict[str, Union[float, pd.DataFrame, np.ndarray]]]:
    """
    Analyze the data for all wells in the dataset.

    This function calls get_dsf_curve_features for each well position in the dataset
    and returns a dictionary of features for each well.

    Args:
        data (pd.DataFrame): Dataset containing multiple wells
        selected_wells (list[str], optional): List of wells to analyze, if None all wells are
        analyzed
        min_temp (float, optional): Minimum temperature for analysis range
        max_temp (float, optional): Maximum temperature for analysis range
        smoothing (float, default=0.01): Smoothing factor for spline fitting
        avg_control_tm (float, optional): Average Tm of control wells

    Returns:
        Dict[str, Dict]: Dictionary where keys are well positions and values are
                        the feature dictionaries returned by get_dsf_curve_features
    """

    if selected_wells is None:
        well_positions = data["well_position"].unique()
    else:
        well_positions = selected_wells

    all_wells_features = {}

    for well in well_positions:
        well_data = data.loc[data["well_position"] == well, :].copy()

        try:
            well_features = get_dsf_curve_features(
                well_data,
                min_temp=min_temp,
                max_temp=max_temp,
                smoothing=smoothing,
                avg_control_tm=avg_control_tm,
            )
            all_wells_features[well] = well_features
        except Exception as e:
            print(f"Error processing well {well}: {str(e)}")
            continue

    return all_wells_features
