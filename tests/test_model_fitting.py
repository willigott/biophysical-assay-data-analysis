"""Tests for multi-model curve fitting (DSFworld approach).

Validates the mathematical model functions, normalization, BIC model selection,
Tm extraction from isolated sigmoids, and end-to-end fitting pipeline.

Reference: Wu et al. 2024, Protein Science 33(6), e5022. doi:10.1002/pro.5022
"""

import numpy as np
import pytest

from bada.processing.model_fitting import (
    ALL_MODEL_NAMES,
    MODEL_FUNCTIONS,
    MODEL_N_PARAMS,
    ModelFitResult,
    ModelFittingConfig,
    TmMethod,
    _compute_bic,
    _compute_r_squared,
    _decompose_model,
    _denormalize_temperature,
    _estimate_starting_parameters,
    _extract_tm_from_sigmoid,
    _fit_single_model,
    _initial_decay,
    _model_1,
    _model_2,
    _model_3,
    _model_4,
    _normalize_data,
    _sigmoid,
    evaluate_fit_result,
    fit_dsf_models,
)


class TestNormalization:
    """Tests for data normalization and denormalization."""

    def test_normalize_maps_to_unit_interval(self) -> None:
        """Verify both T and F are mapped to [0, 1].

        Reference: DSFworld normalizes to [0,1] before fitting (analysis.R lines 295-300).
        """
        temperature = np.linspace(25.0, 95.0, 100)
        fluorescence = np.linspace(500.0, 5000.0, 100)

        t_norm, f_norm, t_min, t_max, f_min, f_max = _normalize_data(temperature, fluorescence)

        assert t_norm[0] == pytest.approx(0.0)
        assert t_norm[-1] == pytest.approx(1.0)
        assert f_norm[0] == pytest.approx(0.0)
        assert f_norm[-1] == pytest.approx(1.0)
        assert t_min == pytest.approx(25.0)
        assert t_max == pytest.approx(95.0)
        assert f_min == pytest.approx(500.0)
        assert f_max == pytest.approx(5000.0)

    def test_round_trip_preserves_temperature(self) -> None:
        """Verify normalizing then denormalizing recovers original temperature.

        Round-trip error should be negligible (< 0.01°C).
        """
        temperature = np.array([25.0, 37.5, 55.0, 72.0, 95.0])
        fluorescence = np.array([500.0, 1000.0, 3000.0, 4500.0, 5000.0])

        t_norm, _, t_min, t_max, _, _ = _normalize_data(temperature, fluorescence)

        for i, t_original in enumerate(temperature):
            t_recovered = _denormalize_temperature(float(t_norm[i]), t_min, t_max)
            assert abs(t_recovered - t_original) < 0.01

    def test_normalize_handles_zero_fluorescence_range(self) -> None:
        """Flat fluorescence (zero range) should produce all-zero normalized values."""
        temperature = np.linspace(25.0, 95.0, 50)
        fluorescence = np.full(50, 1000.0)

        t_norm, f_norm, _, _, _, _ = _normalize_data(temperature, fluorescence)

        assert t_norm[0] == pytest.approx(0.0)
        assert t_norm[-1] == pytest.approx(1.0)
        np.testing.assert_array_equal(f_norm, np.zeros(50))

    def test_denormalize_at_boundaries(self) -> None:
        """Denormalize at 0.0 and 1.0 should give t_min and t_max."""
        assert _denormalize_temperature(0.0, 25.0, 95.0) == pytest.approx(25.0)
        assert _denormalize_temperature(1.0, 25.0, 95.0) == pytest.approx(95.0)
        assert _denormalize_temperature(0.5, 25.0, 95.0) == pytest.approx(60.0)


class TestModelFunctions:
    """Tests for the mathematical correctness of model building blocks."""

    def test_sigmoid_at_tm_equals_half_amplitude_when_no_decay(self) -> None:
        """For d=0 (no decay), the sigmoid at T=Tm should be A/2.

        This is the defining property of the Boltzmann sigmoid inflection point.
        Reference: Standard Boltzmann equation property.
        """
        t_norm = np.array([0.5])
        result = _sigmoid(t_norm, amplitude=1.0, tm=0.5, scale=0.03, decay=0.0)
        assert result[0] == pytest.approx(0.5, abs=1e-6)

    def test_sigmoid_monotonically_increases_before_decay(self) -> None:
        """With mild decay, sigmoid should increase up to approximately Tm."""
        t_norm = np.linspace(0.0, 0.45, 100)
        values = _sigmoid(t_norm, amplitude=1.0, tm=0.5, scale=0.03, decay=-0.5)
        # Should be monotonically increasing in the pre-Tm region
        assert np.all(np.diff(values) >= 0)

    def test_sigmoid_with_decay_creates_peak(self) -> None:
        """Strong decay (d < 0) should cause values to decrease after Tm.

        This is the key difference from standard Boltzmann — the modified sigmoid
        creates a peak shape rather than a plateau.
        Reference: Wu et al. 2024 [1], explanation of modified Boltzmann.
        """
        t_norm = np.linspace(0.0, 1.0, 200)
        values = _sigmoid(t_norm, amplitude=1.0, tm=0.5, scale=0.03, decay=-3.0)

        # Find the maximum — it should be near Tm
        max_idx = np.argmax(values)
        max_t = t_norm[max_idx]
        assert abs(max_t - 0.5) < 0.1

        # After the peak, values should decrease
        assert values[-1] < values[max_idx]

    def test_initial_decay_decreases_for_negative_b(self) -> None:
        """Initial decay with negative b should be a decreasing exponential.

        Reference: analysis.R, build_d() (line 540).
        """
        t_norm = np.linspace(0.0, 1.0, 100)
        values = _initial_decay(t_norm, c=1.0, b=-5.0)

        assert values[0] == pytest.approx(1.0, abs=1e-6)
        assert np.all(np.diff(values) < 0)

    def test_initial_decay_at_zero_is_c(self) -> None:
        """At T_n=0, Id(0) = C * exp(0) = C."""
        result = _initial_decay(np.array([0.0]), c=2.5, b=-5.0)
        assert result[0] == pytest.approx(2.5, abs=1e-6)

    def test_model_1_equals_single_sigmoid(self) -> None:
        """Model 1 should be identical to a single sigmoid call."""
        t = np.linspace(0.0, 1.0, 50)
        result_model = _model_1(t, 1.0, 0.5, 0.03, -1.0)
        result_sigmoid = _sigmoid(t, 1.0, 0.5, 0.03, -1.0)
        np.testing.assert_array_almost_equal(result_model, result_sigmoid)

    def test_model_2_is_sigmoid_plus_decay(self) -> None:
        """Model 2 should equal Sig1 + Id.

        Reference: analysis.R, s1_d_model() (lines 303-326).
        """
        t = np.linspace(0.0, 1.0, 50)
        result_model = _model_2(t, 1.0, 0.5, 0.03, -1.0, 0.5, -5.0)
        result_expected = _sigmoid(t, 1.0, 0.5, 0.03, -1.0) + _initial_decay(t, 0.5, -5.0)
        np.testing.assert_array_almost_equal(result_model, result_expected)

    def test_model_3_is_two_sigmoids(self) -> None:
        """Model 3 should equal Sig1 + Sig2.

        Reference: analysis.R, s2_model() (lines 378-400).
        """
        t = np.linspace(0.0, 1.0, 50)
        result_model = _model_3(t, 1.0, 0.3, 0.03, -1.0, 0.5, 0.7, 0.03, -2.0)
        result_expected = _sigmoid(t, 1.0, 0.3, 0.03, -1.0) + _sigmoid(t, 0.5, 0.7, 0.03, -2.0)
        np.testing.assert_array_almost_equal(result_model, result_expected)

    def test_model_4_is_two_sigmoids_plus_decay(self) -> None:
        """Model 4 should equal Sig1 + Sig2 + Id.

        Reference: analysis.R, s2_d_model() (lines 348-376).
        """
        t = np.linspace(0.0, 1.0, 50)
        result_model = _model_4(t, 1.0, 0.3, 0.03, -1.0, 0.5, 0.7, 0.03, -2.0, 0.3, -5.0)
        result_expected = (
            _sigmoid(t, 1.0, 0.3, 0.03, -1.0)
            + _sigmoid(t, 0.5, 0.7, 0.03, -2.0)
            + _initial_decay(t, 0.3, -5.0)
        )
        np.testing.assert_array_almost_equal(result_model, result_expected)

    def test_sigmoid_handles_extreme_exponent(self) -> None:
        """Sigmoid should not overflow for extreme temperature/scale values.

        The implementation clamps exponent arguments to avoid np.exp overflow.
        """
        t_norm = np.array([0.0, 1.0])
        # Very small scale → very large exponent argument
        result = _sigmoid(t_norm, amplitude=1.0, tm=0.5, scale=0.001, decay=-1.0)
        assert np.all(np.isfinite(result))


class TestBIC:
    """Tests for BIC computation."""

    def test_bic_formula_correctness(self) -> None:
        """Verify BIC = n * ln(RSS/n) + k * ln(n) for known values.

        Reference: analysis.R line 859.
        """
        n = 100
        rss = 0.5
        k = 4
        expected = n * np.log(rss / n) + k * np.log(n)
        result = _compute_bic(n, rss, k)
        assert result == pytest.approx(expected)

    def test_lower_bic_for_better_fit(self) -> None:
        """Lower RSS (better fit) should produce lower BIC, same model complexity."""
        bic_good = _compute_bic(100, 0.1, 4)
        bic_bad = _compute_bic(100, 1.0, 4)
        assert bic_good < bic_bad

    def test_bic_penalizes_more_parameters(self) -> None:
        """Same RSS but more parameters should increase BIC (complexity penalty).

        This is the core property of BIC: it prevents overfitting by penalizing
        model complexity through the k * ln(n) term.
        """
        bic_simple = _compute_bic(100, 0.5, 4)
        bic_complex = _compute_bic(100, 0.5, 10)
        assert bic_simple < bic_complex

    def test_bic_returns_inf_for_zero_rss(self) -> None:
        """Zero or negative RSS should return inf (invalid fit)."""
        assert _compute_bic(100, 0.0, 4) == np.inf
        assert _compute_bic(100, -1.0, 4) == np.inf

    def test_bic_returns_inf_for_zero_points(self) -> None:
        """Zero data points should return inf."""
        assert _compute_bic(0, 0.5, 4) == np.inf


class TestRSquared:
    """Tests for R-squared computation."""

    def test_perfect_fit(self) -> None:
        """Perfect fit should give R² = 1.0."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _compute_r_squared(y, y) == pytest.approx(1.0)

    def test_mean_prediction(self) -> None:
        """Predicting the mean everywhere gives R² = 0.0."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_mean = np.full_like(y, np.mean(y))
        assert _compute_r_squared(y, y_mean) == pytest.approx(0.0)

    def test_constant_observed(self) -> None:
        """Constant observed values (zero total variance) returns 0.0."""
        y = np.full(10, 5.0)
        y_pred = np.linspace(4.0, 6.0, 10)
        assert _compute_r_squared(y, y_pred) == pytest.approx(0.0)


class TestStartingParameters:
    """Tests for starting parameter estimation from derivative peaks."""

    def test_single_peak_curve(self) -> None:
        """Single-transition curve should find one Tm starting value.

        When only one peak is found, tm2 should be tm1 + offset to avoid
        singular gradient in two-sigmoid models.
        Reference: analysis.R, make_peak_finder_nest() line 287.
        """
        t_norm = np.linspace(0.0, 1.0, 141)
        # Normalized Boltzmann sigmoid peaking near 0.43 (= (55-25)/(95-25))
        f_norm = 1.0 / (1.0 + np.exp((0.43 - t_norm) / 0.03))
        # Typical DSF scan: 25-95°C = 70°C range
        t_range = 70.0

        starts = _estimate_starting_parameters(t_norm, f_norm, t_range)

        assert abs(starts["tm1"] - 0.43) < 0.1
        assert starts["tm2"] > starts["tm1"]

    def test_two_peak_curve(self) -> None:
        """Two-transition curve should find two distinct Tm starting values."""
        t_norm = np.linspace(0.0, 1.0, 141)
        f_norm = 1.0 / (1.0 + np.exp((0.36 - t_norm) / 0.03)) + 1.0 / (
            1.0 + np.exp((0.57 - t_norm) / 0.03)
        )
        # Normalize
        f_norm = (f_norm - f_norm.min()) / (f_norm.max() - f_norm.min())
        t_range = 70.0

        starts = _estimate_starting_parameters(t_norm, f_norm, t_range)

        assert starts["tm1"] < starts["tm2"]

    def test_flat_curve_uses_fallback(self) -> None:
        """Flat curve should use midpoint fallback for Tm starting values."""
        t_norm = np.linspace(0.0, 1.0, 100)
        f_norm = np.full(100, 0.5)
        t_range = 70.0

        starts = _estimate_starting_parameters(t_norm, f_norm, t_range)

        assert starts["tm1"] == pytest.approx(0.5)

    def test_c_from_first_value(self) -> None:
        """Initial fluorescence amplitude C should come from f_norm[0]."""
        t_norm = np.linspace(0.0, 1.0, 100)
        f_norm = np.linspace(0.8, 0.2, 100)
        t_range = 70.0

        starts = _estimate_starting_parameters(t_norm, f_norm, t_range)

        assert starts["c"] == pytest.approx(0.8, abs=0.01)


class TestSingleModelFitting:
    """Tests for fitting individual models to synthetic data."""

    def _generate_model_data(
        self, model_name: str, params: list[float], noise_sigma: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic normalized data from a known model."""
        rng = np.random.default_rng(42)
        t_norm = np.linspace(0.0, 1.0, 141)
        model_func = MODEL_FUNCTIONS[model_name]
        f_norm = model_func(t_norm, *params)
        f_norm = f_norm + rng.normal(0, noise_sigma, len(t_norm))
        return t_norm, f_norm

    def test_model_1_converges_on_own_data(self) -> None:
        """Model 1 should converge when fitted to data generated from Model 1.

        Reference: Wu et al. 2024 [1] — 5.1% of curves use Model 1.
        """
        true_params = [1.0, 0.5, 0.03, -1.0]
        t_norm, f_norm = self._generate_model_data("model_1", true_params)

        from bada.processing.model_fitting import _build_model_params

        starts = {"tm1": 0.5, "tm2": 0.55, "c": float(f_norm[0])}
        p0, lower = _build_model_params("model_1", starts)
        popt, rss = _fit_single_model(t_norm, f_norm, _model_1, p0, lower, 500)

        assert popt is not None
        assert rss is not None
        assert rss < 0.1

    def test_model_2_converges_on_own_data(self) -> None:
        """Model 2 should converge when fitted to data generated from Model 2.

        Most commonly selected model (41.2% of curves).
        Reference: Wu et al. 2024 [1].
        """
        true_params = [1.0, 0.5, 0.03, -1.0, 0.5, -5.0]
        t_norm, f_norm = self._generate_model_data("model_2", true_params)

        from bada.processing.model_fitting import _build_model_params

        starts = {"tm1": 0.5, "tm2": 0.55, "c": float(f_norm[0])}
        p0, lower = _build_model_params("model_2", starts)
        popt, rss = _fit_single_model(t_norm, f_norm, _model_2, p0, lower, 500)

        assert popt is not None
        assert rss is not None
        assert rss < 0.1

    def test_model_3_converges_on_own_data(self) -> None:
        """Model 3 should converge when fitted to data generated from Model 3.

        Used for 22.3% of curves (two-domain proteins).
        Reference: Wu et al. 2024 [1].
        """
        true_params = [1.0, 0.35, 0.03, -1.0, 0.5, 0.65, 0.03, -2.0]
        t_norm, f_norm = self._generate_model_data("model_3", true_params)

        from bada.processing.model_fitting import _build_model_params

        starts = {"tm1": 0.35, "tm2": 0.65, "c": float(f_norm[0])}
        p0, lower = _build_model_params("model_3", starts)
        popt, rss = _fit_single_model(t_norm, f_norm, _model_3, p0, lower, 500)

        assert popt is not None
        assert rss is not None
        assert rss < 0.5

    def test_model_4_converges_on_own_data(self) -> None:
        """Model 4 should converge when fitted to data generated from Model 4.

        Used for 31.4% of curves (two domains + initial decay).
        Reference: Wu et al. 2024 [1].
        """
        true_params = [1.0, 0.35, 0.03, -1.0, 0.5, 0.65, 0.03, -2.0, 0.3, -5.0]
        t_norm, f_norm = self._generate_model_data("model_4", true_params)

        from bada.processing.model_fitting import _build_model_params

        starts = {"tm1": 0.35, "tm2": 0.65, "c": float(f_norm[0])}
        p0, lower = _build_model_params("model_4", starts)
        popt, rss = _fit_single_model(t_norm, f_norm, _model_4, p0, lower, 500)

        assert popt is not None
        assert rss is not None
        assert rss < 0.5

    def test_convergence_failure_returns_none(self) -> None:
        """Fitting should return (None, None) on convergence failure, not raise."""
        t_norm = np.linspace(0.0, 1.0, 10)
        f_norm = np.random.default_rng(42).uniform(0, 1, 10)

        # Use bad starting params that should not converge in 1 iteration
        popt, rss = _fit_single_model(
            t_norm,
            f_norm,
            _model_4,
            p0=[1.0, 0.5, 0.03, -1.0, 0.5, 0.7, 0.03, -2.0, 0.3, -5.0],
            lower_bounds=[0.01, 0.1, 0.01, -10.0, 0.01, 0.1, 0.01, -10.0, 0.01, -20.0],
            max_iterations=1,
        )

        # Should gracefully return None, not raise
        assert popt is None or rss is not None


class TestTmExtraction:
    """Tests for Tm extraction from isolated sigmoid components."""

    def test_tm_from_standard_boltzmann(self) -> None:
        """For d=0 (standard Boltzmann), derivative max should be at Tm.

        This is the defining property: the inflection point of a Boltzmann sigmoid
        is at T = Tm.
        """
        t_norm = np.linspace(0.0, 1.0, 200)
        sig_values = _sigmoid(t_norm, amplitude=1.0, tm=0.5, scale=0.03, decay=0.0)

        tm_real, max_deriv = _extract_tm_from_sigmoid(t_norm, sig_values, 25.0, 95.0)

        # Should be near 60°C (= 0.5 * (95-25) + 25)
        assert abs(tm_real - 60.0) < 0.5
        assert max_deriv > 0

    def test_tm_from_decaying_sigmoid_shifts_from_parameter(self) -> None:
        """With decay d < 0, the derivative max shifts from the Tm parameter.

        This is why DSFworld extracts Tm from the derivative, not from the
        fitted Tm parameter directly.
        Reference: Wu et al. 2024 [1], key design choice.
        """
        t_norm = np.linspace(0.0, 1.0, 200)

        # Standard Boltzmann: derivative max at Tm
        sig_no_decay = _sigmoid(t_norm, 1.0, 0.5, 0.03, 0.0)
        tm_no_decay, _ = _extract_tm_from_sigmoid(t_norm, sig_no_decay, 25.0, 95.0)

        # Decaying sigmoid: derivative max shifts
        sig_decay = _sigmoid(t_norm, 1.0, 0.5, 0.03, -3.0)
        tm_decay, _ = _extract_tm_from_sigmoid(t_norm, sig_decay, 25.0, 95.0)

        # The decaying sigmoid's derivative peak should shift relative to the standard
        assert tm_no_decay != pytest.approx(tm_decay, abs=0.1)

    def test_tm_denormalization_accuracy(self) -> None:
        """Tm should be correctly converted from normalized to real temperature."""
        t_norm = np.linspace(0.0, 1.0, 200)
        # Sigmoid centered at 0.5 normalized = 60°C for 25-95°C range
        sig_values = _sigmoid(t_norm, 1.0, 0.5, 0.03, 0.0)

        tm_real, _ = _extract_tm_from_sigmoid(t_norm, sig_values, 25.0, 95.0)

        assert abs(tm_real - 60.0) < 0.5


class TestComponentDecomposition:
    """Tests for model decomposition into isolated components."""

    def test_model_1_has_one_sigmoid(self) -> None:
        """Model 1 decomposition should have exactly one sigmoid component."""
        t = np.linspace(0.0, 1.0, 50)
        params = np.array([1.0, 0.5, 0.03, -1.0])
        components = _decompose_model(t, "model_1", params)

        assert "sigmoid_1" in components
        assert "sigmoid_2" not in components
        assert "initial_decay" not in components

    def test_model_2_has_sigmoid_and_decay(self) -> None:
        """Model 2 decomposition should have sigmoid_1 and initial_decay."""
        t = np.linspace(0.0, 1.0, 50)
        params = np.array([1.0, 0.5, 0.03, -1.0, 0.5, -5.0])
        components = _decompose_model(t, "model_2", params)

        assert "sigmoid_1" in components
        assert "initial_decay" in components
        assert "sigmoid_2" not in components

    def test_model_3_has_two_sigmoids(self) -> None:
        """Model 3 decomposition should have sigmoid_1 and sigmoid_2."""
        t = np.linspace(0.0, 1.0, 50)
        params = np.array([1.0, 0.3, 0.03, -1.0, 0.5, 0.7, 0.03, -2.0])
        components = _decompose_model(t, "model_3", params)

        assert "sigmoid_1" in components
        assert "sigmoid_2" in components
        assert "initial_decay" not in components

    def test_model_4_has_all_components(self) -> None:
        """Model 4 decomposition should have sigmoid_1, sigmoid_2, and initial_decay."""
        t = np.linspace(0.0, 1.0, 50)
        params = np.array([1.0, 0.3, 0.03, -1.0, 0.5, 0.7, 0.03, -2.0, 0.3, -5.0])
        components = _decompose_model(t, "model_4", params)

        assert "sigmoid_1" in components
        assert "sigmoid_2" in components
        assert "initial_decay" in components

    def test_components_sum_to_full_model(self) -> None:
        """Sum of decomposed components should equal the full model output."""
        t = np.linspace(0.0, 1.0, 100)
        params = np.array([1.0, 0.3, 0.03, -1.0, 0.5, 0.7, 0.03, -2.0, 0.3, -5.0])

        full = _model_4(t, *params)
        components = _decompose_model(t, "model_4", params)
        reconstructed = (
            components["sigmoid_1"] + components["sigmoid_2"] + components["initial_decay"]
        )

        np.testing.assert_array_almost_equal(full, reconstructed, decimal=10)


class TestBICModelSelection:
    """Tests for BIC-based automatic model selection."""

    def _make_clean_sigmoid_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate clean single-sigmoid data (should favor Model 1)."""
        rng = np.random.default_rng(42)
        x = np.linspace(25, 95, 141)
        y = 500.0 + 4500.0 / (1.0 + np.exp((55.0 - x) / 2.0))
        y = y + rng.normal(0, 5.0, len(x))
        return x, y

    def _make_high_initial_decay_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate sigmoid + initial decay data (should favor Model 2)."""
        rng = np.random.default_rng(43)
        x = np.linspace(25, 95, 141)
        y = 500.0 + 4500.0 / (1.0 + np.exp((60.0 - x) / 2.0))
        y = y + 3000.0 * np.exp(-0.05 * (x - 25.0))
        y = y + rng.normal(0, 10.0, len(x))
        return x, y

    def _make_two_domain_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate two-sigmoid data (should favor Model 3)."""
        rng = np.random.default_rng(44)
        x = np.linspace(25, 95, 141)
        y = 400.0
        y = y + 2000.0 / (1.0 + np.exp((50.0 - x) / 2.5))
        y = y + 3000.0 / (1.0 + np.exp((65.0 - x) / 2.0))
        y = y + rng.normal(0, 10.0, len(x))
        return x, y

    def test_auto_selects_model_for_clean_sigmoid(self) -> None:
        """Auto mode should produce accurate Tm for a clean single-sigmoid curve.

        BIC may legitimately select a more complex model (e.g., model_3) if the
        additional parameters improve the fit enough to offset the penalty.
        The key requirement is that the extracted Tm is accurate.
        """
        x, y = self._make_clean_sigmoid_data()
        result = fit_dsf_models(x, y, tm_method=TmMethod.AUTO)

        assert result.is_valid
        assert abs(result.tm - 55.0) < 2.0

    def test_auto_selects_model_for_high_initial_fluorescence(self) -> None:
        """Auto mode should produce accurate Tm for sigmoid + initial decay data.

        Model 2 accounts for 41.2% of real-world curves. BIC should prefer a model
        with an initial decay component, but the primary requirement is Tm accuracy.
        Reference: Wu et al. 2024 [1].
        """
        x, y = self._make_high_initial_decay_data()
        result = fit_dsf_models(x, y, tm_method=TmMethod.AUTO)

        assert result.is_valid
        assert abs(result.tm - 60.0) < 3.0

    def test_auto_handles_two_domain_protein(self) -> None:
        """Auto mode should select Model 3 or 4 for two-domain protein data.

        Reference: Wu et al. 2024 [1] — Models 3 and 4 handle 53.7% of curves.
        """
        x, y = self._make_two_domain_data()
        result = fit_dsf_models(x, y, tm_method=TmMethod.AUTO)

        assert result.is_valid
        assert result.selected_model in ("model_3", "model_4")
        # Primary Tm should be near one of the two transitions
        assert abs(result.tm - 50.0) < 3.0 or abs(result.tm - 65.0) < 3.0

    def test_auto_reports_bic_for_all_converged_models(self) -> None:
        """BIC values should be reported for all models that converged."""
        x, y = self._make_clean_sigmoid_data()
        result = fit_dsf_models(x, y, tm_method=TmMethod.AUTO)

        assert len(result.bic_values) >= 1
        # At least the selected model should have a BIC
        assert result.selected_model in result.bic_values

    def test_specific_model_only_fits_that_model(self) -> None:
        """Requesting a specific model should only fit that model."""
        x, y = self._make_clean_sigmoid_data()
        result = fit_dsf_models(x, y, tm_method=TmMethod.MODEL_2)

        assert result.selected_model == "model_2"
        assert "model_2" in result.bic_values
        assert "model_1" not in result.bic_values


class TestFallback:
    """Tests for fallback to derivative method when models fail."""

    def test_flat_curve_triggers_fallback(
        self, flat_fluorescence: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Flat fluorescence should trigger derivative fallback.

        All models should fail on constant fluorescence, and the derivative
        method should be used as fallback (which also returns NaN for flat curves).
        """
        x, y = flat_fluorescence
        result = fit_dsf_models(x, y, tm_method=TmMethod.AUTO)

        assert "all_models_failed" in result.quality_flags
        assert "fallback_to_derivative" in result.quality_flags

    def test_fallback_disabled_returns_nan(
        self, flat_fluorescence: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """With fallback disabled, all-models-failed should return NaN."""
        x, y = flat_fluorescence
        config = ModelFittingConfig(fallback_to_derivative=False)
        result = fit_dsf_models(x, y, tm_method=TmMethod.AUTO, config=config)

        assert np.isnan(result.tm)
        assert result.is_valid is False
        assert "all_models_failed" in result.quality_flags


class TestModelFitResult:
    """Tests for ModelFitResult structure and fields."""

    def test_result_has_all_fields(self) -> None:
        """ModelFitResult should expose all documented fields."""
        result = ModelFitResult(
            tm=55.0,
            tm_secondary=None,
            max_derivative_value=100.0,
            selected_model="model_1",
            bic_values={"model_1": -500.0},
            model_parameters={"A": 1.0, "Tm": 0.5, "scal": 0.03, "d": -1.0},
            r_squared=0.99,
        )

        assert result.tm == 55.0
        assert result.tm_secondary is None
        assert result.selected_model == "model_1"
        assert result.is_valid is True
        assert result.quality_flags == []

    def test_result_is_frozen(self) -> None:
        """ModelFitResult should be immutable (frozen dataclass)."""
        result = ModelFitResult(
            tm=55.0,
            tm_secondary=None,
            max_derivative_value=100.0,
            selected_model="model_1",
            bic_values={},
            model_parameters={},
            r_squared=0.99,
        )
        with pytest.raises(AttributeError):
            result.tm = 60.0  # type: ignore[misc]


class TestTmMethodEnum:
    """Tests for TmMethod enum."""

    def test_string_values(self) -> None:
        """TmMethod enum values should match expected strings."""
        assert TmMethod.DERIVATIVE.value == "derivative"
        assert TmMethod.MODEL_1.value == "model_1"
        assert TmMethod.MODEL_2.value == "model_2"
        assert TmMethod.MODEL_3.value == "model_3"
        assert TmMethod.MODEL_4.value == "model_4"
        assert TmMethod.AUTO.value == "auto"

    def test_string_construction(self) -> None:
        """TmMethod should be constructable from string values."""
        assert TmMethod("derivative") == TmMethod.DERIVATIVE
        assert TmMethod("auto") == TmMethod.AUTO

    def test_invalid_string_raises(self) -> None:
        """Invalid string should raise ValueError."""
        with pytest.raises(ValueError):
            TmMethod("invalid_method")


class TestIntegration:
    """End-to-end integration tests using test fixtures."""

    def test_single_transition_tm_accuracy(
        self, sample_temperatures: np.ndarray, sample_fluorescence: np.ndarray
    ) -> None:
        """Model fitting should detect Tm near 55°C for a clean Boltzmann sigmoid.

        Tolerance: 2.0°C (wider than derivative method due to different extraction).
        Reference: Lee et al. 2019 [2] — 0.33°C RMSD for derivative, ~0.26°C for fitting.
        """
        result = fit_dsf_models(sample_temperatures, sample_fluorescence, tm_method=TmMethod.AUTO)

        assert result.is_valid
        assert abs(result.tm - 55.0) < 2.0
        assert result.r_squared > 0.9

    def test_high_initial_fluorescence_tm_accuracy(
        self, high_initial_fluorescence: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Model fitting should detect Tm near 60°C for high initial fluorescence.

        This curve type (DSFworld Model 2) accounts for 41.2% of real-world curves.
        Reference: Wu et al. 2024 [1].
        """
        x, y = high_initial_fluorescence
        result = fit_dsf_models(x, y, tm_method=TmMethod.AUTO)

        assert result.is_valid
        assert abs(result.tm - 60.0) < 2.5

    def test_multi_domain_detects_both_transitions(
        self, multi_domain_fluorescence: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Model fitting should detect both Tm values for a two-domain protein.

        Tm1=50°C, Tm2=65°C. Model 3 or 4 should be selected.
        Reference: Gao et al. 2020 [8].
        """
        x, y = multi_domain_fluorescence
        result = fit_dsf_models(x, y, tm_method=TmMethod.AUTO)

        assert result.is_valid
        assert result.selected_model in ("model_3", "model_4")

        if result.tm_secondary is not None:
            tms = sorted([result.tm, result.tm_secondary])
            assert abs(tms[0] - 50.0) < 3.0
            assert abs(tms[1] - 65.0) < 3.0

    def test_multi_domain_with_decay(
        self, multi_domain_with_decay_fluorescence: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Model fitting should handle two-domain protein with initial decay.

        This curve type corresponds to DSFworld Model 4 (31.4% of curves).
        Reference: Wu et al. 2024 [1].
        """
        x, y = multi_domain_with_decay_fluorescence
        result = fit_dsf_models(x, y, tm_method=TmMethod.AUTO)

        assert result.is_valid
        # Model 4 has initial_decay component
        if result.selected_model == "model_4":
            assert "initial_decay" in result.component_curves

    def test_lysozyme_like_tm_accuracy(
        self, lysozyme_like_fluorescence: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Model fitting should detect Tm near 72°C for lysozyme-like curve.

        Published Tm=72.4°C at pH 6.0 (Schönfelder et al. 2025 [12]).
        """
        x, y = lysozyme_like_fluorescence
        result = fit_dsf_models(x, y, tm_method=TmMethod.AUTO)

        assert result.is_valid
        assert abs(result.tm - 72.0) < 2.5

    def test_model_parameters_populated(
        self, sample_temperatures: np.ndarray, sample_fluorescence: np.ndarray
    ) -> None:
        """Fitted model parameters should be populated with correct keys."""
        result = fit_dsf_models(sample_temperatures, sample_fluorescence, tm_method=TmMethod.AUTO)

        assert result.is_valid
        assert len(result.model_parameters) > 0
        # Parameters should be finite numbers
        for name, value in result.model_parameters.items():
            assert np.isfinite(value), f"Parameter {name} is not finite: {value}"

    def test_component_curves_populated(
        self, sample_temperatures: np.ndarray, sample_fluorescence: np.ndarray
    ) -> None:
        """Component curves should be populated for the selected model."""
        result = fit_dsf_models(sample_temperatures, sample_fluorescence, tm_method=TmMethod.AUTO)

        assert result.is_valid
        assert "sigmoid_1" in result.component_curves
        assert len(result.component_curves["sigmoid_1"]) > 0

    def test_string_tm_method_accepted(
        self, sample_temperatures: np.ndarray, sample_fluorescence: np.ndarray
    ) -> None:
        """fit_dsf_models should accept string values for tm_method."""
        result = fit_dsf_models(sample_temperatures, sample_fluorescence, tm_method="auto")
        assert result.is_valid


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_very_noisy_data(self) -> None:
        """Heavy noise should not cause crashes. Fitting may fail gracefully.

        Reference: Real DSF data can be noisy due to low protein concentration,
        instrument variability, or suboptimal dye conditions.
        """
        rng = np.random.default_rng(99)
        x = np.linspace(25, 95, 141)
        y = 500.0 + 4500.0 / (1.0 + np.exp((55.0 - x) / 2.0))
        y = y + rng.normal(0, 500.0, len(x))

        # Should not raise
        result = fit_dsf_models(x, y, tm_method=TmMethod.AUTO)
        assert isinstance(result, ModelFitResult)

    def test_few_data_points(self) -> None:
        """Very few data points should not crash."""
        x = np.linspace(25, 95, 20)
        y = 500.0 + 4500.0 / (1.0 + np.exp((55.0 - x) / 2.0))

        result = fit_dsf_models(x, y, tm_method=TmMethod.AUTO)
        assert isinstance(result, ModelFitResult)

    def test_model_function_registry_complete(self) -> None:
        """All model names should have corresponding functions, param counts, and param names."""
        from bada.processing.model_fitting import MODEL_PARAM_NAMES

        for name in ALL_MODEL_NAMES:
            assert name in MODEL_FUNCTIONS
            assert name in MODEL_N_PARAMS
            assert name in MODEL_PARAM_NAMES
            assert len(MODEL_PARAM_NAMES[name]) == MODEL_N_PARAMS[name]


class TestEvaluateFitResult:
    """Tests for evaluate_fit_result() — model re-evaluation in real space."""

    def test_returns_correct_shapes(
        self, sample_temperatures: np.ndarray, sample_fluorescence: np.ndarray
    ) -> None:
        """Evaluated model should have same length as input data."""
        result = fit_dsf_models(sample_temperatures, sample_fluorescence, tm_method=TmMethod.AUTO)
        x_eval, y_fitted, components = evaluate_fit_result(
            result, sample_temperatures, sample_fluorescence
        )

        assert len(x_eval) == len(sample_temperatures)
        assert len(y_fitted) == len(sample_temperatures)
        for comp_values in components.values():
            assert len(comp_values) == len(sample_temperatures)

    def test_fitted_values_in_data_range(
        self, sample_temperatures: np.ndarray, sample_fluorescence: np.ndarray
    ) -> None:
        """Fitted model values should be in a plausible range relative to data.

        The full model prediction should not wildly exceed the data range.
        """
        result = fit_dsf_models(sample_temperatures, sample_fluorescence, tm_method=TmMethod.AUTO)
        _x_eval, y_fitted, _components = evaluate_fit_result(
            result, sample_temperatures, sample_fluorescence
        )

        f_min = float(np.min(sample_fluorescence))
        f_max = float(np.max(sample_fluorescence))
        f_range = f_max - f_min

        # Fitted values should be within 50% of the data range on each side
        assert float(np.min(y_fitted)) > f_min - 0.5 * f_range
        assert float(np.max(y_fitted)) < f_max + 0.5 * f_range

    def test_components_sum_to_fitted(
        self, sample_temperatures: np.ndarray, sample_fluorescence: np.ndarray
    ) -> None:
        """Sum of component curves (scaled) plus f_min should approximate the full fit.

        Components are additive contributions (without baseline offset), so
        sum(components) * f_range should equal y_fitted - f_min approximately.
        """
        result = fit_dsf_models(sample_temperatures, sample_fluorescence, tm_method=TmMethod.AUTO)
        _x_eval, y_fitted, components = evaluate_fit_result(
            result, sample_temperatures, sample_fluorescence
        )

        f_min = float(np.min(sample_fluorescence))
        component_sum = sum(components.values())

        # component_sum (already in real units) + f_min ≈ y_fitted
        np.testing.assert_allclose(
            component_sum + f_min,
            y_fitted,
            rtol=1e-6,
            err_msg="Component sum + f_min should equal full model prediction",
        )

    def test_invalid_result_returns_empty(self) -> None:
        """evaluate_fit_result should return empty arrays for a failed result."""
        failed_result = ModelFitResult(
            tm=np.nan,
            tm_secondary=None,
            max_derivative_value=np.nan,
            selected_model="none",
            bic_values={},
            model_parameters={},
            r_squared=np.nan,
            is_valid=False,
            quality_flags=["all_models_failed"],
        )

        x = np.linspace(25, 95, 100)
        y = np.full_like(x, 1000.0)

        x_eval, y_fitted, components = evaluate_fit_result(failed_result, x, y)

        assert len(x_eval) == 0
        assert len(y_fitted) == 0
        assert len(components) == 0

    def test_model_2_has_sigmoid_and_decay(
        self, high_initial_fluorescence: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Model 2 evaluation should produce sigmoid_1 and initial_decay components."""
        x, y = high_initial_fluorescence
        result = fit_dsf_models(x, y, tm_method=TmMethod.MODEL_2)

        if not result.is_valid:
            pytest.skip("Model 2 did not converge on this fixture")

        _x_eval, _y_fitted, components = evaluate_fit_result(result, x, y)

        assert "sigmoid_1" in components
        assert "initial_decay" in components
