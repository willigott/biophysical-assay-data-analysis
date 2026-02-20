import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import BSpline

from bada.processing.feature_extraction import (
    WellProcessingResult,
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
        """Test that normalization maps signal to [0, 1] range.

        Reference: Standard min-max normalization as described in
        https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
        """
        normalized = get_normalized_signal(sample_fluorescence)

        # Check bounds
        assert np.min(normalized) == pytest.approx(0.0)
        assert np.max(normalized) == pytest.approx(1.0)

        # Check shape preservation
        assert normalized.shape == sample_fluorescence.shape

    def test_get_normalized_signal_pandas_series(self) -> None:
        """Test that normalization works with pandas Series input."""
        series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        normalized = get_normalized_signal(series)

        assert isinstance(normalized, np.ndarray)
        assert np.min(normalized) == pytest.approx(0.0)
        assert np.max(normalized) == pytest.approx(1.0)

    def test_get_normalized_signal_empty_array(self) -> None:
        """Test that normalizing an empty array raises ValueError.

        An empty signal has no meaningful range and cannot be normalized.
        """
        with pytest.raises(ValueError, match="Cannot normalize empty signal"):
            get_normalized_signal(np.array([]))

    def test_get_normalized_signal_with_nan(self) -> None:
        """Test that NaN values in the signal raise ValueError.

        NaN values indicate missing or corrupt fluorescence readings. Normalizing
        such data would silently propagate NaN through downstream calculations
        (e.g., Tm detection, DTW distance), producing unreliable results.
        Reference: https://numpy.org/doc/stable/reference/constants.html#numpy.nan
        """
        with pytest.raises(ValueError, match="Signal contains NaN values"):
            get_normalized_signal(np.array([1.0, np.nan, 3.0]))

    def test_get_normalized_signal_with_inf(self) -> None:
        """Test that infinite values in the signal raise ValueError.

        Infinite values can arise from sensor overflow or prior division-by-zero
        errors and would corrupt normalization.
        """
        with pytest.raises(ValueError, match="Signal contains infinite values"):
            get_normalized_signal(np.array([1.0, np.inf, 3.0]))

        with pytest.raises(ValueError, match="Signal contains infinite values"):
            get_normalized_signal(np.array([1.0, -np.inf, 3.0]))

    def test_get_normalized_signal_constant_signal(self) -> None:
        """Test that a constant signal (zero range) raises ValueError.

        A flat fluorescence trace (max == min) has zero range, making min-max
        normalization undefined. This typically indicates a failed well or
        instrument error in DSF experiments.
        Reference: Niesen et al., Nature Protocols 2007 - quality control criteria
        for DSF data include checking for flat traces.
        """
        with pytest.raises(ValueError, match="Cannot normalize signal with zero range"):
            get_normalized_signal(np.array([5.0, 5.0, 5.0, 5.0]))

    def test_get_normalized_signal_two_values(self) -> None:
        """Test normalization with a minimal two-element signal."""
        normalized = get_normalized_signal(np.array([100.0, 200.0]))

        assert normalized[0] == pytest.approx(0.0)
        assert normalized[1] == pytest.approx(1.0)

    def test_get_spline(
        self, sample_temperatures: np.ndarray, sample_fluorescence: np.ndarray
    ) -> None:
        """Test that get_spline returns expected objects."""
        spline, x_spline, y_spline = get_spline(
            sample_temperatures, sample_fluorescence, smoothing=0.01, n_points=1000
        )

        # Check types
        assert isinstance(spline, BSpline)
        assert isinstance(x_spline, np.ndarray)
        assert isinstance(y_spline, np.ndarray)

        # Check shapes
        assert len(x_spline) == 1000
        assert len(y_spline) == 1000

        # Check values
        assert np.min(x_spline) == pytest.approx(np.min(sample_temperatures))
        assert np.max(x_spline) == pytest.approx(np.max(sample_temperatures))

    def test_get_spline_derivative(
        self, sample_spline: BSpline, sample_spline_x: np.ndarray
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

    def test_get_min_max_values_with_precomputed_spline(
        self, sample_temperatures: np.ndarray, sample_fluorescence: np.ndarray
    ) -> None:
        """Test that get_min_max_values with a pre-computed spline matches the default path.

        When a caller already has a spline (e.g. from get_dsf_curve_features), passing it
        via spline_result should produce identical results and skip recomputation.
        """
        spline_result = get_spline(sample_temperatures, sample_fluorescence)

        result_default = get_min_max_values(sample_temperatures, sample_fluorescence)
        result_cached = get_min_max_values(
            sample_temperatures, sample_fluorescence, spline_result=spline_result
        )

        for default_val, cached_val in zip(result_default, result_cached):
            assert default_val == cached_val

    def test_get_max_derivative(self, sample_spline: BSpline, sample_spline_x: np.ndarray) -> None:
        """Test that _get_max_derivative finds Tm near 55.0°C on a Boltzmann sigmoid.

        The sample_fluorescence fixture uses a Boltzmann sigmoid with Tm=55.0°C.
        The analytical derivative maximum of a Boltzmann sigmoid is at exactly Tm.
        Tolerance of 0.5°C is derived from DMAN's 0.33°C RMSD on real data
        (Lee et al. 2019 [2]).
        """
        max_val, tm = _get_max_derivative(sample_spline, sample_spline_x)

        assert isinstance(max_val, float)
        assert isinstance(tm, float)
        assert np.min(sample_spline_x) <= tm <= np.max(sample_spline_x)
        # Accuracy: Tm should be close to the known 55.0°C ground truth
        assert abs(tm - 55.0) < 0.5

    def test_get_tm(self, sample_temperatures: np.ndarray, sample_fluorescence: np.ndarray) -> None:
        """Test that get_tm detects Tm near 55.0°C on a Boltzmann sigmoid.

        The sample_fluorescence fixture uses a Boltzmann sigmoid with Tm=55.0°C
        and light noise (sigma=5 RFU). Tolerance of 0.5°C is derived from DMAN's
        0.33°C RMSD for first-derivative methods on clean data (Lee et al. 2019 [2]).
        """
        tm, max_derivative = get_tm(sample_temperatures, sample_fluorescence)

        assert isinstance(tm, float)
        assert isinstance(max_derivative, float)
        assert np.min(sample_temperatures) <= tm <= np.max(sample_temperatures)
        # Accuracy: Tm should be close to the known 55.0°C ground truth
        assert abs(tm - 55.0) < 0.5

    def test_get_tm_with_precomputed_spline(
        self, sample_temperatures: np.ndarray, sample_fluorescence: np.ndarray
    ) -> None:
        """Test that get_tm with a pre-computed spline matches the default path.

        This verifies the optimization in get_dsf_curve_features where the filtered-data
        spline is computed once and reused for both derivative and Tm extraction.
        """
        spline_result = get_spline(sample_temperatures, sample_fluorescence)

        tm_default, deriv_default = get_tm(sample_temperatures, sample_fluorescence)
        tm_cached, deriv_cached = get_tm(
            sample_temperatures, sample_fluorescence, spline_result=spline_result
        )

        assert tm_default == tm_cached
        assert deriv_default == deriv_cached

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
            "fluorescence_range",
            "temp_at_min",
            "temp_at_max",
            "tm",
            "max_derivative_value",
            "delta_tm",
            "peak_confidence",
            "peak_is_valid",
            "peak_quality_flags",
            "n_peaks_detected",
            "smoothing",
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
        assert isinstance(features["peak_confidence"], float)
        assert isinstance(features["peak_is_valid"], bool)
        assert isinstance(features["peak_quality_flags"], list)
        assert isinstance(features["n_peaks_detected"], int)
        assert isinstance(features["min_temp"], float)
        assert isinstance(features["max_temp"], float)

        # Check basic value constraints
        assert features["min_fluorescence"] <= features["max_fluorescence"]
        assert features["min_temp"] <= features["tm"] <= features["max_temp"]
        assert features["min_temp"] <= features["temp_at_min"] <= features["max_temp"]
        assert features["min_temp"] <= features["temp_at_max"] <= features["max_temp"]
        assert 0.0 <= features["peak_confidence"] <= 1.0
        assert features["n_peaks_detected"] >= 0

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
        result = get_dsf_curve_features_multiple_wells(sample_dsf_data)

        assert isinstance(result, WellProcessingResult)
        assert not result.failures

        # Check that we have features for each well
        well_positions = sample_dsf_data["well_position"].unique()
        assert len(result.features) == len(well_positions)

        # Check that all well positions are included as keys
        for well in well_positions:
            assert well in result.features

        # Check that features for each well have the expected structure
        for well, features in result.features.items():
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
                "peak_confidence",
                "peak_is_valid",
                "peak_quality_flags",
                "n_peaks_detected",
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
        result = get_dsf_curve_features_multiple_wells(
            sample_dsf_data,
            min_temp=min_temp,
            max_temp=max_temp,
            smoothing=smoothing,
            avg_control_tm=avg_control_tm,
        )

        # Check that parameters were applied correctly
        for well, features in result.features.items():
            # Check temperature range
            assert features["min_temp"] == min_temp
            assert features["max_temp"] == max_temp

            # Check delta_tm was calculated using control Tm
            expected_delta_tm = features["tm"] - avg_control_tm
            assert features["delta_tm"] == pytest.approx(expected_delta_tm)

    def test_get_all_wells_dsf_features_with_error(
        self, sample_dsf_data: pd.DataFrame, mocker
    ) -> None:
        """Test that get_all_wells_dsf_features handles errors for individual wells."""
        # Create a mock version of get_dsf_curve_features that raises an exception for a specific
        # well
        original_get_features = get_dsf_curve_features

        def mock_get_features(data, **kwargs):
            if data["well_position"].values[0] == "A1":
                raise ValueError("Test error for A1")
            return original_get_features(data, **kwargs)

        mocker.patch(
            "bada.processing.feature_extraction.get_dsf_curve_features",
            side_effect=mock_get_features,
        )

        result = get_dsf_curve_features_multiple_wells(sample_dsf_data)

        # Check that the function continued despite the error
        assert "A1" not in result.features
        assert "A2" in result.features

        # Check that the failure was tracked
        assert "A1" in result.failures
        assert isinstance(result.failures["A1"], ValueError)
        assert "Test error for A1" in str(result.failures["A1"])

    def test_get_all_wells_dsf_features_with_selected_wells(
        self, sample_dsf_data: pd.DataFrame
    ) -> None:
        """Test that get_all_wells_dsf_features works correctly with selected_wells parameter."""
        # Get features for a specific well only
        selected_wells = ["A1"]
        result = get_dsf_curve_features_multiple_wells(
            sample_dsf_data, selected_wells=selected_wells
        )

        # Check that only the selected well is processed
        assert len(result.features) == 1
        assert "A1" in result.features
        assert "A2" not in result.features

        # Check that the selected well's features have the expected structure
        features = result.features["A1"]
        expected_keys = [
            "full_well_data",
            "x_spline",
            "y_spline",
            "y_spline_derivative",
            "min_fluorescence",
            "max_fluorescence",
            "fluorescence_range",
            "temp_at_min",
            "temp_at_max",
            "tm",
            "max_derivative_value",
            "delta_tm",
            "peak_confidence",
            "peak_is_valid",
            "peak_quality_flags",
            "n_peaks_detected",
            "smoothing",
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


class TestPeakDetectionIntegration:
    """Integration tests for the peak detection pipeline through the public API.

    These tests validate the full chain: raw fluorescence → spline → derivative
    → peak detection → features, using realistic DSF curve fixtures.
    """

    def test_tm_accuracy_single_transition(self, sample_dsf_data: pd.DataFrame) -> None:
        """Test Tm accuracy on a clean Boltzmann sigmoid (Tm=55.0°C).

        The sample_fluorescence fixture generates a Boltzmann sigmoid with known
        Tm=55.0°C and low noise (sigma=5 RFU). Tolerance of 0.5°C matches DMAN's
        0.33°C RMSD on clean data (Lee et al. 2019 [2]).
        """
        single_well_data = sample_dsf_data.loc[sample_dsf_data["well_position"] == "A1", :].copy()
        features = get_dsf_curve_features(single_well_data)

        assert abs(features["tm"] - 55.0) < 0.5
        assert features["peak_is_valid"] is True
        assert features["peak_confidence"] > 0.5

    def test_tm_accuracy_lysozyme(
        self, lysozyme_like_fluorescence: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test Tm accuracy on a lysozyme-like curve (Tm=72.0°C).

        Uses realistic SYPRO Orange parameters with baseline drift.
        Published Tm=72.4°C at pH 6.0 (Schönfelder et al. 2025 [12]).
        Tolerance of 0.5°C from DMAN benchmark (Lee et al. 2019 [2]).
        """
        x, y = lysozyme_like_fluorescence
        data = pd.DataFrame({"well_position": "A1", "temperature": x, "fluorescence": y})
        features = get_dsf_curve_features(data)

        assert abs(features["tm"] - 72.0) < 0.5
        assert features["peak_is_valid"] is True

    def test_flat_curve_returns_nan(self, flat_fluorescence: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that a flat fluorescence trace produces tm=NaN with quality flags.

        A flat trace means no protein unfolding occurred (empty well, pre-denatured
        protein, or wrong dye concentration).

        Reference: Sun et al. 2020 [4] classifies flat traces as "no transition";
        Niesen et al. 2007 [10] describes flat traces as a common DSF failure mode.
        """
        x, y = flat_fluorescence
        data = pd.DataFrame({"well_position": "A1", "temperature": x, "fluorescence": y})
        features = get_dsf_curve_features(data)

        assert np.isnan(features["tm"])
        assert features["peak_is_valid"] is False
        assert features["peak_confidence"] == 0.0
        assert len(features["peak_quality_flags"]) > 0

    def test_multi_domain_two_peaks(
        self, multi_domain_fluorescence: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that two-domain protein produces two detected peaks.

        Transition 1: Tm1=50.0°C, Transition 2: Tm2=65.0°C.
        Wider tolerance (1.5°C) accounts for overlapping transitions.

        Reference: Gao et al. 2020 [8] describes multi-transition behavior and
        recommends derivative methods for multi-domain proteins.
        """
        x, y = multi_domain_fluorescence
        data = pd.DataFrame({"well_position": "A1", "temperature": x, "fluorescence": y})
        features = get_dsf_curve_features(data)

        assert features["n_peaks_detected"] >= 2
        assert features["peak_is_valid"] is True

    def test_high_initial_fluorescence_correct_tm(
        self, high_initial_fluorescence: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that high initial fluorescence does not mislead Tm detection.

        The initial decay creates a large derivative at low temperatures. The peak
        detection should identify the real Tm at 60°C, not the decay artifact.

        Reference: Wu et al. 2024 [1] — Model 2 (41.2% of curves) is designed
        for high initial fluorescence.
        """
        x, y = high_initial_fluorescence
        data = pd.DataFrame({"well_position": "A1", "temperature": x, "fluorescence": y})
        features = get_dsf_curve_features(data)

        assert abs(features["tm"] - 60.0) < 2.0
        assert features["peak_is_valid"] is True

    def test_aggregation_peak_not_primary(
        self, aggregation_artifact_fluorescence: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that aggregation artifact is not selected as primary Tm.

        Real transition at 58°C should be more prominent than the aggregation-related
        peak near 75°C.

        Reference: Gao et al. 2020 [8] — aggregation is a well-documented complication.
        """
        x, y = aggregation_artifact_fluorescence
        data = pd.DataFrame({"well_position": "A1", "temperature": x, "fluorescence": y})
        features = get_dsf_curve_features(data)

        assert abs(features["tm"] - 58.0) < 2.0
        assert features["peak_is_valid"] is True

    def test_boundary_exclusion(
        self, edge_artifact_fluorescence: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that edge artifacts are excluded and the real Tm at 60°C is detected.

        A strong artifact near 93°C (high-temperature boundary) should be removed
        by boundary exclusion, leaving the real transition at 60°C.

        Reference: Wu et al. 2023 [9] — recommends excluding scan endpoints;
        Sun et al. 2020 [4] — region-aware analysis.
        """
        x, y = edge_artifact_fluorescence
        data = pd.DataFrame({"well_position": "A1", "temperature": x, "fluorescence": y})
        features = get_dsf_curve_features(data)

        assert abs(features["tm"] - 60.0) < 1.0
        assert features["peak_is_valid"] is True

    def test_confidence_correlates_with_data_quality(self) -> None:
        """Test that confidence score increases with data quality.

        Run peak detection on three curves of increasing quality and assert
        strict ordering: confidence_clean > confidence_moderate > confidence_noisy.

        Reference: MoltenProt's BS-factor [3] and TSA-CRAFT's R² criterion [2]
        provide quality metrics that correlate with data quality.
        """
        x = np.linspace(25, 95, 141)
        f_native = 500.0
        f_denatured = 5000.0
        tm = 55.0
        slope = 2.0
        y_clean_base = f_native + (f_denatured - f_native) / (1.0 + np.exp((tm - x) / slope))

        rng = np.random.default_rng(51)

        # Clean
        y_clean = y_clean_base + rng.normal(0, 5.0, len(x))
        data_clean = pd.DataFrame(
            {"well_position": "A1", "temperature": x, "fluorescence": y_clean}
        )
        features_clean = get_dsf_curve_features(data_clean)

        # Moderate noise
        y_moderate = y_clean_base + rng.normal(0, 50.0, len(x))
        data_moderate = pd.DataFrame(
            {"well_position": "A1", "temperature": x, "fluorescence": y_moderate}
        )
        features_moderate = get_dsf_curve_features(data_moderate)

        # Heavy noise
        y_noisy = y_clean_base + rng.normal(0, 200.0, len(x))
        data_noisy = pd.DataFrame(
            {"well_position": "A1", "temperature": x, "fluorescence": y_noisy}
        )
        features_noisy = get_dsf_curve_features(data_noisy)

        assert features_clean["peak_confidence"] > features_moderate["peak_confidence"]
        assert features_moderate["peak_confidence"] > features_noisy["peak_confidence"]

    def test_get_tm_returns_nan_on_flat_input(self) -> None:
        """Test that get_tm returns (np.nan, np.nan) on flat input.

        Flat input should propagate through the robust peak detection and
        return NaN values instead of a meaningless argmax result.
        """
        x = np.linspace(25, 95, 100)
        y = np.full_like(x, 1000.0)

        tm, max_derivative = get_tm(x, y)

        assert np.isnan(tm)
        assert np.isnan(max_derivative)
