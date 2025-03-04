import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from .preprocessing import get_spline, get_spline_derivative


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


def _get_max_derivative(spline: UnivariateSpline, x_spline: np.ndarray) -> tuple[float, float]:
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

    return tm, max_derivative_value
