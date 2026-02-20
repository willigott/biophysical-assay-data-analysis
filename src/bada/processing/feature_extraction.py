from dataclasses import dataclass, field
import logging
from typing import TypedDict

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from bada.processing.peak_detection import PeakDetectionConfig, detect_melting_peaks
from bada.processing.preprocessing import SplineResult, get_spline, get_spline_derivative
from bada.utils.validation import validate_temperature_range

logger = logging.getLogger(__name__)


class DSFCurveFeatures(TypedDict):
    """Features extracted from a single DSF melting curve.

    Returned by get_dsf_curve_features(). Keys fall into four groups:

    Raw/fitted data:
        full_well_data: Complete raw well data (unfiltered).
        x_spline: Temperature values for spline fit (within analysis range).
        y_spline: Fluorescence values for spline fit.
        y_spline_derivative: First derivative of spline (dF/dT).

    Scalar features:
        min_fluorescence: Min fluorescence from full curve spline.
        max_fluorescence: Max fluorescence from full curve spline.
        fluorescence_range: max_fluorescence - min_fluorescence.
        temp_at_min: Temperature at min fluorescence.
        temp_at_max: Temperature at max fluorescence.
        tm: Melting temperature (temperature at max first derivative).
        max_derivative_value: Peak value of first derivative.
        delta_tm: tm - avg_control_tm; np.nan if no control provided.

    Peak detection quality:
        peak_confidence: Confidence score in [0.0, 1.0] for the detected Tm.
        peak_is_valid: Whether a valid peak was found and the curve is not problematic.
        peak_quality_flags: Quality issue descriptions (e.g., "no_valid_peak",
            "flat_derivative", "too_many_peaks").
        n_peaks_detected: Number of valid peaks detected after filtering.

    Analysis parameters:
        smoothing: Smoothing parameter used for spline fitting.
        min_temp: Lower bound of temperature analysis range.
        max_temp: Upper bound of temperature analysis range.
    """

    full_well_data: pd.DataFrame
    x_spline: np.ndarray
    y_spline: np.ndarray
    y_spline_derivative: np.ndarray
    min_fluorescence: float
    max_fluorescence: float
    fluorescence_range: float
    temp_at_min: float
    temp_at_max: float
    tm: float
    max_derivative_value: float
    delta_tm: float
    peak_confidence: float
    peak_is_valid: bool
    peak_quality_flags: list[str]
    n_peaks_detected: int
    smoothing: float
    min_temp: float
    max_temp: float


@dataclass
class WellProcessingResult:
    """Result of processing multiple wells.

    Attributes:
        features: Mapping of well position to feature dict for successfully processed wells.
        failures: Mapping of well position to exception for wells that failed processing.
    """

    features: dict[str, DSFCurveFeatures] = field(default_factory=dict)
    failures: dict[str, Exception] = field(default_factory=dict)


def get_min_max_values(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    spline_result: SplineResult | None = None,
    **kwargs,
) -> tuple[float, float, float, float]:
    """Get minimum and maximum values from spline fit

    Args:
        x: Temperature values.
        y: Fluorescence values.
        spline_result: Pre-computed spline result from get_spline(). If None, a new
            spline is fitted using x, y, and any additional kwargs.

    Returns:
        tuple[float, float, float, float]: (min_value_y, max_value_y, x_at_min, x_at_max)
    """
    if spline_result is None:
        _, x_spline, y_spline = get_spline(x, y, **kwargs)
    else:
        _, x_spline, y_spline = spline_result

    min_idx = np.argmin(y_spline)
    max_idx = np.argmax(y_spline)

    y_min = y_spline[min_idx]
    y_max = y_spline[max_idx]

    x_at_min_y = x_spline[min_idx]
    x_at_max_y = x_spline[max_idx]

    return (y_min, y_max, x_at_min_y, x_at_max_y)


def _get_max_derivative(
    spline: BSpline,
    x_spline: np.ndarray,
    config: PeakDetectionConfig | None = None,
) -> tuple[float, float]:
    """Get maximum derivative from spline fit using robust peak detection.

    Delegates to detect_melting_peaks() for validated peak selection instead
    of naive np.argmax(). Returns (np.nan, np.nan) when no valid peak is found.

    Args:
        spline: Fitted B-spline.
        x_spline: Temperature values for the spline.
        config: Peak detection parameters. Uses defaults if None.

    Returns:
        tuple[float, float]: (max_derivative_value, x_at_max_derivative)
    """
    y_spline_derivative = get_spline_derivative(spline, x_spline)
    result = detect_melting_peaks(x_spline, y_spline_derivative, config=config)

    return (result.max_derivative_value, result.tm)


def get_tm(
    temperature: np.ndarray | pd.Series,
    fluorescence: np.ndarray | pd.Series,
    spline_result: SplineResult | None = None,
    **kwargs,
) -> tuple[float, float]:
    """Get melting temperature (Tm) from signal

    Args:
        temperature: Temperature values.
        fluorescence: Fluorescence values.
        spline_result: Pre-computed spline result from get_spline(). If None, a new
            spline is fitted using temperature, fluorescence, and any additional kwargs.

    Returns:
        tuple[float, float]: (tm, max_derivative_value)
    """
    if spline_result is None:
        spline, x_spline, _ = get_spline(temperature, fluorescence, **kwargs)
    else:
        spline, x_spline, _ = spline_result
    max_derivative_value, tm = _get_max_derivative(spline, x_spline)

    return (tm, max_derivative_value)


def get_dsf_curve_features(
    data: pd.DataFrame,
    min_temp: float | None = None,
    max_temp: float | None = None,
    smoothing: float = 0.01,
    avg_control_tm: float | None = None,
    peak_detection_config: PeakDetectionConfig | None = None,
) -> DSFCurveFeatures:
    """
    Analyze the data for a single well.

    Args:
        data (pd.DataFrame): Dataset pre-filtered for a single well
        min_temp (float, optional): Minimum temperature for analysis range
        max_temp (float, optional): Maximum temperature for analysis range
        smoothing (float, default=0.01): Smoothing factor for spline fitting
        avg_control_tm (float, optional): Average Tm of control wells
        peak_detection_config (PeakDetectionConfig, optional): Configuration for peak
            detection. Uses defaults if None.

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

    filtered_spline_result = get_spline(
        np.asarray(filtered_data["temperature"]),
        np.asarray(filtered_data["fluorescence"]),
        smoothing=smoothing,
    )
    spline, x_spline, y_spline = filtered_spline_result

    y_spline_derivative = np.asarray(spline.derivative()(x_spline))

    peak_result = detect_melting_peaks(x_spline, y_spline_derivative, config=peak_detection_config)

    temp_at_max_derivative = peak_result.tm
    max_derivative_value = peak_result.max_derivative_value

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
        "peak_confidence": peak_result.peak_confidence,
        "peak_is_valid": peak_result.is_valid,
        "peak_quality_flags": peak_result.quality_flags,
        "n_peaks_detected": peak_result.n_peaks_found,
        "smoothing": smoothing,
        "min_temp": min_temp,
        "max_temp": max_temp,
    }


def get_dsf_curve_features_multiple_wells(
    data: pd.DataFrame,
    selected_wells: list[str] | None = None,
    min_temp: float | None = None,
    max_temp: float | None = None,
    smoothing: float = 0.01,
    avg_control_tm: float | None = None,
    peak_detection_config: PeakDetectionConfig | None = None,
) -> WellProcessingResult:
    """
    Analyze the data for all wells in the dataset.

    This function calls get_dsf_curve_features for each well position in the dataset
    and returns a result containing features for successful wells and errors for failed wells.

    Args:
        data (pd.DataFrame): Dataset containing multiple wells
        selected_wells (list[str], optional): List of wells to analyze, if None all wells are
        analyzed
        min_temp (float, optional): Minimum temperature for analysis range
        max_temp (float, optional): Maximum temperature for analysis range
        smoothing (float, default=0.01): Smoothing factor for spline fitting
        avg_control_tm (float, optional): Average Tm of control wells
        peak_detection_config (PeakDetectionConfig, optional): Configuration for peak
            detection. Uses defaults if None.

    Returns:
        WellProcessingResult: Contains .features (successful wells) and .failures (failed wells)
    """

    if selected_wells is None:
        well_positions = data["well_position"].unique()
    else:
        well_positions = selected_wells

    result = WellProcessingResult()

    for well in well_positions:
        well_data = data.loc[data["well_position"] == well, :].copy()

        try:
            well_features = get_dsf_curve_features(
                well_data,
                min_temp=min_temp,
                max_temp=max_temp,
                smoothing=smoothing,
                avg_control_tm=avg_control_tm,
                peak_detection_config=peak_detection_config,
            )
            result.features[well] = well_features
        except (ValueError, np.linalg.LinAlgError, TypeError) as e:
            result.failures[well] = e
            logger.warning("Failed to process well %s: %s", well, e)

    n_total = len(well_positions)
    n_failed = len(result.failures)
    logger.info(
        "Processed %d wells: %d succeeded, %d failed", n_total, n_total - n_failed, n_failed
    )

    return result
