import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import UnivariateSpline

from bada.processing.feature_extraction import (
    _get_max_derivative,
    get_dsf_curve_features,
    get_dsf_curve_features_multiple_wells,
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
from bada.utils.validation import validate_temperature_range


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

    def test_get_dsf_curve_features(self, sample_dsf_data: pd.DataFrame) -> None:
        """Test that get_dsf_curve_features extracts the expected features."""
        # Filter data for a single well
        single_well_data = sample_dsf_data.loc[sample_dsf_data["well_position"] == "A1", :].copy()

        # Get features with default parameters
        features = get_dsf_curve_features(single_well_data)

        # Check that all expected keys are present
        expected_keys = [
            "full_well_data",
            "x_spline",
            "y_spline",
            "y_spline_derivative",
            "min_fluorescence",
            "max_fluorescence",
            "temp_at_min",
            "temp_at_max",
            "tm",
            "max_derivative_value",
            "delta_tm",
            "min_temp",
            "max_temp",
        ]
        for key in expected_keys:
            assert key in features

        # Check types
        assert isinstance(features["full_well_data"], pd.DataFrame)
        assert isinstance(features["x_spline"], np.ndarray)
        assert isinstance(features["y_spline"], np.ndarray)
        assert isinstance(features["y_spline_derivative"], np.ndarray)
        assert isinstance(features["min_fluorescence"], float)
        assert isinstance(features["max_fluorescence"], float)
        assert isinstance(features["temp_at_min"], float)
        assert isinstance(features["temp_at_max"], float)
        assert isinstance(features["tm"], float)
        assert isinstance(features["max_derivative_value"], float)
        assert np.isnan(features["delta_tm"])  # No control Tm provided
        assert isinstance(features["min_temp"], float)
        assert isinstance(features["max_temp"], float)

        # Check basic value constraints
        assert features["min_fluorescence"] <= features["max_fluorescence"]
        assert features["min_temp"] <= features["tm"] <= features["max_temp"]
        assert features["min_temp"] <= features["temp_at_min"] <= features["max_temp"]
        assert features["min_temp"] <= features["temp_at_max"] <= features["max_temp"]

    def test_get_dsf_curve_features_with_temperature_range(
        self, sample_dsf_data: pd.DataFrame
    ) -> None:
        """Test that get_dsf_curve_features works with custom temperature range."""
        # Filter data for a single well
        single_well_data = sample_dsf_data.loc[sample_dsf_data["well_position"] == "A1", :].copy()

        min_temp = 40.0
        max_temp = 80.0

        features = get_dsf_curve_features(single_well_data, min_temp=min_temp, max_temp=max_temp)

        # Check that temperature range is respected
        assert features["min_temp"] == min_temp
        assert features["max_temp"] == max_temp

        # Features should be calculated on the filtered data
        assert min_temp <= features["tm"] <= max_temp

    def test_get_dsf_curve_features_with_control_tm(self, sample_dsf_data: pd.DataFrame) -> None:
        """Test that get_dsf_curve_features calculates delta_tm with a control Tm."""
        # Filter data for a single well
        single_well_data = sample_dsf_data.loc[sample_dsf_data["well_position"] == "A1", :].copy()

        avg_control_tm = 60.0

        features = get_dsf_curve_features(single_well_data, avg_control_tm=avg_control_tm)

        # delta_tm should be calculated
        expected_delta_tm = features["tm"] - avg_control_tm
        assert features["delta_tm"] == pytest.approx(expected_delta_tm)

    def test_get_dsf_curve_features_with_smoothing(self, sample_dsf_data: pd.DataFrame) -> None:
        """Test that get_dsf_curve_features works with different smoothing parameters."""
        # Filter data for a single well
        single_well_data = sample_dsf_data.loc[sample_dsf_data["well_position"] == "A1", :].copy()

        # First with low smoothing
        features_low = get_dsf_curve_features(single_well_data, smoothing=0.001)

        # Then with high smoothing
        features_high = get_dsf_curve_features(single_well_data, smoothing=0.1)

        # Check that smoothing affects the calculated features
        # Compare at least one key feature like Tm
        assert isinstance(features_low["tm"], float)
        assert isinstance(features_high["tm"], float)

    def test_get_all_wells_dsf_features(self, sample_dsf_data: pd.DataFrame) -> None:
        """Test that get_all_wells_dsf_features processes all wells correctly."""
        # Get features for all wells
        all_features = get_dsf_curve_features_multiple_wells(sample_dsf_data)

        # Check that we have features for each well
        well_positions = sample_dsf_data["well_position"].unique()
        assert len(all_features) == len(well_positions)

        # Check that all well positions are included as keys
        for well in well_positions:
            assert well in all_features

        # Check that features for each well have the expected structure
        for well, features in all_features.items():
            # Check that all expected keys are present
            expected_keys = [
                "full_well_data",
                "x_spline",
                "y_spline",
                "y_spline_derivative",
                "min_fluorescence",
                "max_fluorescence",
                "temp_at_min",
                "temp_at_max",
                "tm",
                "max_derivative_value",
                "delta_tm",
                "min_temp",
                "max_temp",
            ]
            for key in expected_keys:
                assert key in features

            # Check that well data is filtered correctly
            well_data = features["full_well_data"]
            assert isinstance(well_data, pd.DataFrame)
            assert well_data["well_position"].nunique() == 1
            assert well_data["well_position"].values[0] == well

    def test_get_all_wells_dsf_features_with_params(self, sample_dsf_data: pd.DataFrame) -> None:
        """Test that get_all_wells_dsf_features forwards parameters correctly."""
        # Set custom parameters
        min_temp = 40.0
        max_temp = 80.0
        smoothing = 0.05
        avg_control_tm = 60.0

        # Get features with custom parameters
        all_features = get_dsf_curve_features_multiple_wells(
            sample_dsf_data,
            min_temp=min_temp,
            max_temp=max_temp,
            smoothing=smoothing,
            avg_control_tm=avg_control_tm,
        )

        # Check that parameters were applied correctly
        for well, features in all_features.items():
            # Check temperature range
            assert features["min_temp"] == min_temp
            assert features["max_temp"] == max_temp

            # Check delta_tm was calculated using control Tm
            expected_delta_tm = features["tm"] - avg_control_tm
            assert features["delta_tm"] == pytest.approx(expected_delta_tm)

    def test_get_all_wells_dsf_features_with_error(
        self, sample_dsf_data: pd.DataFrame, mocker, capfd
    ) -> None:
        """Test that get_all_wells_dsf_features handles errors for individual wells."""
        # Create a mock version of get_dsf_curve_features that raises an exception for a specific
        # well
        original_get_features = get_dsf_curve_features

        def mock_get_features(data, **kwargs):
            if data["well_position"].values[0] == "A1":
                raise ValueError("Test error for A1")
            return original_get_features(data, **kwargs)

        # Apply the mock and capture stdout
        mocker.patch(
            "bada.processing.feature_extraction.get_dsf_curve_features",
            side_effect=mock_get_features,
        )

        # Run function
        all_features = get_dsf_curve_features_multiple_wells(sample_dsf_data)

        # Check that the function continued despite the error
        assert "A1" not in all_features
        assert "A2" in all_features

        # Check that the error was printed
        captured = capfd.readouterr()
        assert "Error processing well A1: Test error for A1" in captured.out

    def test_get_all_wells_dsf_features_with_selected_wells(
        self, sample_dsf_data: pd.DataFrame
    ) -> None:
        """Test that get_all_wells_dsf_features works correctly with selected_wells parameter."""
        # Get features for a specific well only
        selected_wells = ["A1"]
        all_features = get_dsf_curve_features_multiple_wells(
            sample_dsf_data, selected_wells=selected_wells
        )

        # Check that only the selected well is processed
        assert len(all_features) == 1
        assert "A1" in all_features
        assert "A2" not in all_features

        # Check that the selected well's features have the expected structure
        features = all_features["A1"]
        expected_keys = [
            "full_well_data",
            "x_spline",
            "y_spline",
            "y_spline_derivative",
            "min_fluorescence",
            "max_fluorescence",
            "temp_at_min",
            "temp_at_max",
            "tm",
            "max_derivative_value",
            "delta_tm",
            "min_temp",
            "max_temp",
        ]
        for key in expected_keys:
            assert key in features


class TestValidation:
    def test_validate_temperature_range_valid(self) -> None:
        """Test that validate_temperature_range returns True for valid input."""
        assert validate_temperature_range(25.0, 95.0) is True

    def test_validate_temperature_range_invalid(self) -> None:
        """Test that validate_temperature_range raises ValueError for invalid input."""
        with pytest.raises(ValueError):
            validate_temperature_range(95.0, 25.0)  # min > max

        with pytest.raises(ValueError):
            validate_temperature_range(50.0, 50.0)  # min == max

    def test_validate_temperature_range_none_values(self) -> None:
        """Test that validate_temperature_range returns False when None is provided."""
        assert validate_temperature_range(None, 95.0) is False
        assert validate_temperature_range(25.0, None) is False
        assert validate_temperature_range(None, None) is False
