import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import UnivariateSpline

from bada.processing.feature_extraction import (
    _get_max_derivative,
    get_min_max_values,
    get_tm,
)
from bada.processing.preprocessing import (
    _get_dtw_distance_normalized,
    _get_dtw_distance_unnormalized,
    get_dtw_distance,
    get_dtw_distances_from_reference,
    get_normalized_signal,
    get_spline,
    get_spline_derivative,
)


class TestPreprocessing:
    def test_get_normalized_signal(self, sample_fluorescence: np.ndarray) -> None:
        """Test that normalization maps signal to [0, 1] range."""
        normalized = get_normalized_signal(sample_fluorescence)

        # Check bounds
        assert np.min(normalized) == pytest.approx(0.0)
        assert np.max(normalized) == pytest.approx(1.0)

        # Check shape preservation
        assert normalized.shape == sample_fluorescence.shape

    def test_get_spline(
        self, sample_temperatures: np.ndarray, sample_fluorescence: np.ndarray
    ) -> None:
        """Test that get_spline returns expected objects."""
        spline, x_spline, y_spline = get_spline(
            sample_temperatures, sample_fluorescence, smoothing=0.01, n_points=1000
        )

        # Check types
        assert isinstance(spline, UnivariateSpline)
        assert isinstance(x_spline, np.ndarray)
        assert isinstance(y_spline, np.ndarray)

        # Check shapes
        assert len(x_spline) == 1000
        assert len(y_spline) == 1000

        # Check values
        assert np.min(x_spline) == pytest.approx(np.min(sample_temperatures))
        assert np.max(x_spline) == pytest.approx(np.max(sample_temperatures))

    def test_get_spline_derivative(
        self, sample_spline: UnivariateSpline, sample_spline_x: np.ndarray
    ) -> None:
        """Test that get_spline_derivative returns the derivative of the spline."""
        derivative = get_spline_derivative(sample_spline, sample_spline_x)

        # Check type and shape
        assert isinstance(derivative, np.ndarray)
        assert len(derivative) == len(sample_spline_x)

    def test_get_dtw_distance_unnormalized(self, sample_fluorescence: np.ndarray) -> None:
        """Test that _get_dtw_distance_unnormalized calculates distance."""
        # Create a second signal slightly different from the first
        signal2 = sample_fluorescence + np.random.normal(0, 10, len(sample_fluorescence))

        distance = _get_dtw_distance_unnormalized(sample_fluorescence, signal2)

        # Check it returns a float greater than 0
        assert isinstance(distance, float)
        assert distance > 0

    def test_get_dtw_distance_normalized(self, sample_fluorescence: np.ndarray) -> None:
        """Test that _get_dtw_distance_normalized calculates distance after normalization."""
        # Create a scaled and shifted version of the signal
        signal2 = 2 * sample_fluorescence + 100

        # Distance between raw signals should be large
        raw_distance = _get_dtw_distance_unnormalized(sample_fluorescence, signal2)

        # Distance after normalization should be smaller
        norm_distance = _get_dtw_distance_normalized(sample_fluorescence, signal2)

        assert norm_distance < raw_distance

    def test_get_dtw_distance(self, sample_fluorescence: np.ndarray) -> None:
        """Test that get_dtw_distance works in both normalized and unnormalized modes."""
        # Create a scaled and shifted version of the signal
        signal2 = 2 * sample_fluorescence + 100

        # With normalization
        norm_distance = get_dtw_distance(sample_fluorescence, signal2, normalized=True)

        # Without normalization
        raw_distance = get_dtw_distance(sample_fluorescence, signal2, normalized=False)

        assert norm_distance < raw_distance

    def test_get_dtw_distances_from_reference(self, sample_dsf_data: pd.DataFrame) -> None:
        """Test that get_dtw_distances_from_reference returns a dictionary of distances."""
        distances = get_dtw_distances_from_reference(sample_dsf_data, reference_well="A1")

        # Check that the result is a dictionary with expected wells
        assert isinstance(distances, dict)
        assert "A1" in distances
        assert "A2" in distances

        # Check that distance to self is 0
        assert distances["A1"][0] == 0.0

        # Check that reference well is correctly stored
        assert distances["A1"][1] == "A1"
        assert distances["A2"][1] == "A1"

        # Check that distance to other well is greater than 0
        assert distances["A2"][0] > 0


class TestFeatureExtraction:
    def test_get_min_max_values(
        self, sample_temperatures: np.ndarray, sample_fluorescence: np.ndarray
    ) -> None:
        """Test that get_min_max_values returns min and max values."""
        y_min, y_max, x_at_min, x_at_max = get_min_max_values(
            sample_temperatures, sample_fluorescence
        )

        # Should return floats
        assert isinstance(y_min, float)
        assert isinstance(y_max, float)
        assert isinstance(x_at_min, float)
        assert isinstance(x_at_max, float)

        # Min should be less than max
        assert y_min < y_max

        # x values should be within the range of temperatures
        assert np.min(sample_temperatures) <= x_at_min <= np.max(sample_temperatures)
        assert np.min(sample_temperatures) <= x_at_max <= np.max(sample_temperatures)

    def test_get_max_derivative(
        self, sample_spline: UnivariateSpline, sample_spline_x: np.ndarray
    ) -> None:
        """Test that _get_max_derivative finds the maximum of the derivative."""
        max_val, max_idx = _get_max_derivative(sample_spline, sample_spline_x)

        # Should return a float and an index
        assert isinstance(max_val, float)
        assert isinstance(max_idx, float)

        # Index should be within the range of x values
        assert np.min(sample_spline_x) <= max_idx <= np.max(sample_spline_x)

    def test_get_tm(self, sample_temperatures: np.ndarray, sample_fluorescence: np.ndarray) -> None:
        """Test that get_tm calculates the melting temperature."""
        tm, max_derivative = get_tm(sample_temperatures, sample_fluorescence)

        # Should return a float for both values
        assert isinstance(tm, float)
        assert isinstance(max_derivative, float)

        # Tm should be within the range of temperatures
        assert np.min(sample_temperatures) <= tm <= np.max(sample_temperatures)
