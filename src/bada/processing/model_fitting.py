"""Multi-model curve fitting for DSF melting temperature determination.

Implements the DSFworld approach (Wu et al. 2024 [1]) as an alternative Tm detection
method alongside the existing first derivative method. Four parametric models based on
modified Boltzmann sigmoids are fitted to the fluorescence data, and BIC (Bayesian
Information Criterion) selects the best model.

Models:
    1. Single modified sigmoid
    2. Single modified sigmoid + initial fluorescence decay
    3. Two modified sigmoids
    4. Two modified sigmoids + initial fluorescence decay

References:
    [1] Wu et al. 2024, Protein Science 33(6), e5022. doi:10.1002/pro.5022
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from bada.processing.peak_detection import PeakDetectionConfig, detect_melting_peaks
from bada.processing.preprocessing import get_spline, get_spline_derivative

logger = logging.getLogger(__name__)

# Maximum absolute value for exponent arguments passed to np.exp().
# Prevents overflow (exp(710) ≈ inf in float64) while being large enough
# to never affect the actual sigmoid/decay shape in the [0, 1] domain.
_EXP_CLIP_BOUND = 500.0

# --- Starting parameter defaults (from DSFworld analysis.R) ---

# Sigmoid parameters
DEFAULT_AMPLITUDE = 1.0
DEFAULT_SCALE = 0.03
DEFAULT_DECAY = -1.0

# Second sigmoid parameters
DEFAULT_AMPLITUDE_2 = 0.5
DEFAULT_DECAY_2 = -2.0

# Initial fluorescence decay parameters
DEFAULT_DECAY_RATE = -5.0

# --- Bounds (from DSFworld analysis.R) ---

MIN_AMPLITUDE = 0.1
MIN_AMPLITUDE_TWO_SIGMOID = 0.01
MIN_TM = 0.1
MIN_SCALE = 0.01
MIN_DECAY = -10.0
MIN_INITIAL_DECAY_AMPLITUDE = 0.01
MIN_INITIAL_DECAY_RATE = -20.0

# --- Fine grid for Tm extraction ---

TM_EXTRACTION_GRID_POINTS = 1000

# --- Tm offset for singular gradient avoidance ---
# When only one peak is found, create Tm2 = Tm1 + TM2_OFFSET_NORMALIZED
# to avoid singular gradient in two-sigmoid models.
# Reference: analysis.R, make_peak_finder_nest() line 287
TM2_OFFSET_NORMALIZED = 0.05

# --- Minimum Tm separation for two-sigmoid models ---
# Two-domain proteins typically have Tms separated by >= 10 C (e.g., BSA domains
# Tm1 ~ 56 C, Tm2 ~ 70 C). When BIC selects a two-sigmoid model but the two
# extracted Tms are closer than this threshold, the second sigmoid is likely
# absorbing post-transition fluorescence decay rather than capturing a real
# transition. In that case, fall back to the corresponding single-sigmoid model.
# Reference: Niesen et al. 2007, Nature Protocols 2(9), 2212-2221.
MIN_TM_SEPARATION_CELSIUS = 8.0


class TmMethod(str, Enum):
    """Method for melting temperature determination.

    DERIVATIVE: Current BADA approach — B-spline + analytical derivative + find_peaks.
    MODEL_1 through MODEL_4: Specific DSFworld parametric models.
    AUTO: Fit all 4 parametric models, select by BIC, with derivative as fallback.
    """

    DERIVATIVE = "derivative"
    MODEL_1 = "model_1"
    MODEL_2 = "model_2"
    MODEL_3 = "model_3"
    MODEL_4 = "model_4"
    AUTO = "auto"


@dataclass(frozen=True)
class ModelFittingConfig:
    """Configuration for multi-model curve fitting.

    Attributes:
        max_iterations: Maximum number of optimizer iterations. DSFworld uses 500
            (analysis.R line 322).
        fallback_to_derivative: Use derivative method when all models fail to converge.
    """

    max_iterations: int = 500
    fallback_to_derivative: bool = True


@dataclass(frozen=True)
class ModelFitResult:
    """Result of multi-model curve fitting.

    Attributes:
        tm: Primary Tm in real temperature units. np.nan if all fitting fails.
        tm_secondary: Secondary Tm from Sig2 (Models 3/4 only, None otherwise).
        max_derivative_value: Derivative value at Tm of the isolated sigmoid.
        selected_model: Which model was selected (e.g., "model_2").
        bic_values: BIC for each successfully fitted model.
        model_parameters: Fitted parameters of the selected model (normalized space).
        r_squared: R-squared of the selected model fit.
        component_curves: Isolated component curves keyed by "sigmoid_1", "sigmoid_2",
            "initial_decay". Values are arrays evaluated on the normalized temperature grid.
        is_valid: Whether fitting succeeded.
        quality_flags: Quality issues (e.g., "all_models_failed", "fallback_to_derivative").
    """

    tm: float
    tm_secondary: float | None
    max_derivative_value: float
    selected_model: str
    bic_values: dict[str, float]
    model_parameters: dict[str, float]
    r_squared: float
    component_curves: dict[str, np.ndarray] = field(default_factory=dict)
    is_valid: bool = True
    quality_flags: list[str] = field(default_factory=list)


# --- Data normalization ---


def _normalize_data(
    temperature: np.ndarray, fluorescence: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Normalize temperature and fluorescence to [0, 1].

    DSFworld normalizes both axes before fitting so that starting parameters are
    instrument-independent and scale-independent.

    Reference: analysis.R, make_temp_n2r() (lines 295-300), Wu et al. 2024 [1] Appendix S1.

    Args:
        temperature: Raw temperature array in degrees Celsius.
        fluorescence: Raw fluorescence array (RFU).

    Returns:
        Tuple of (t_norm, f_norm, t_min, t_max, f_min, f_max).
    """
    t_min = float(np.min(temperature))
    t_max = float(np.max(temperature))
    f_min = float(np.min(fluorescence))
    f_max = float(np.max(fluorescence))

    t_range = t_max - t_min
    f_range = f_max - f_min

    if np.isclose(t_range, 0.0):
        t_norm = np.zeros_like(temperature, dtype=float)
    else:
        t_norm = (temperature - t_min) / t_range

    if np.isclose(f_range, 0.0):
        f_norm = np.zeros_like(fluorescence, dtype=float)
    else:
        f_norm = (fluorescence - f_min) / f_range

    return t_norm, f_norm, t_min, t_max, f_min, f_max


def _denormalize_temperature(t_norm: float, t_min: float, t_max: float) -> float:
    """Convert normalized temperature back to degrees Celsius.

    Args:
        t_norm: Normalized temperature in [0, 1].
        t_min: Original minimum temperature.
        t_max: Original maximum temperature.

    Returns:
        Temperature in degrees Celsius.
    """
    return t_norm * (t_max - t_min) + t_min


# --- Model building blocks ---


def _sigmoid(
    t_norm: np.ndarray, amplitude: float, tm: float, scale: float, decay: float
) -> np.ndarray:
    """Modified Boltzmann sigmoid with exponential decay factor.

    F(T) = A / (1 + exp((Tm - T) / scal)) * exp(d * (T - Tm))

    The exp(d * (T - Tm)) term causes the sigmoid plateau to decline at temperatures
    above Tm (when d < 0), creating a peak shape rather than a plateau.

    Reference: Wu et al. 2024 [1], analysis.R build_sig1() (line 490).

    Args:
        t_norm: Normalized temperature array in [0, 1].
        amplitude: Scaling factor (A).
        tm: Melting temperature in normalized units.
        scale: Slope control (smaller = steeper transition).
        decay: Post-Tm decay magnitude (typically negative).

    Returns:
        Sigmoid values at each temperature point.
    """
    # Clamp the exponent arguments to avoid overflow
    boltzmann_arg = np.clip((tm - t_norm) / scale, -_EXP_CLIP_BOUND, _EXP_CLIP_BOUND)
    decay_arg = np.clip(decay * (t_norm - tm), -_EXP_CLIP_BOUND, _EXP_CLIP_BOUND)
    return amplitude / (1.0 + np.exp(boltzmann_arg)) * np.exp(decay_arg)


def _initial_decay(t_norm: np.ndarray, c: float, b: float) -> np.ndarray:
    """Exponential initial fluorescence decay component.

    Id(T) = C * exp(T * b)

    Models the exponential decrease in fluorescence at low temperatures, observed when
    protein is partially denatured before the scan, dye binds to hydrophobic surface
    patches, or contamination/lipids are present.

    Reference: Wu et al. 2024 [1], analysis.R build_d() (line 540).

    Args:
        t_norm: Normalized temperature array in [0, 1].
        c: Starting fluorescence value (amplitude of initial fluorescence).
        b: Decay rate (negative; controls how fast initial fluorescence decays).

    Returns:
        Initial decay values at each temperature point.
    """
    decay_arg = np.clip(t_norm * b, -_EXP_CLIP_BOUND, _EXP_CLIP_BOUND)
    return c * np.exp(decay_arg)


# --- The four models ---


def _model_1(t_norm: np.ndarray, a: float, tm: float, scal: float, d: float) -> np.ndarray:
    """Model 1: Single modified sigmoid.

    F(T) = Sig1(T)

    Reference: analysis.R, s1_model() (lines 328-346).
    """
    return _sigmoid(t_norm, a, tm, scal, d)


def _model_2(
    t_norm: np.ndarray, a: float, tm: float, scal: float, d: float, c: float, b: float
) -> np.ndarray:
    """Model 2: Single modified sigmoid + initial fluorescence decay.

    F(T) = Sig1(T) + Id(T)

    Most commonly selected model (41.2% of curves in DSFworld benchmark).

    Reference: analysis.R, s1_d_model() (lines 303-326).
    """
    return _sigmoid(t_norm, a, tm, scal, d) + _initial_decay(t_norm, c, b)


def _model_3(
    t_norm: np.ndarray,
    a1: float,
    tm1: float,
    scal1: float,
    d1: float,
    a2: float,
    tm2: float,
    scal2: float,
    d2: float,
) -> np.ndarray:
    """Model 3: Two modified sigmoids.

    F(T) = Sig1(T) + Sig2(T)

    Reference: analysis.R, s2_model() (lines 378-400).
    """
    return _sigmoid(t_norm, a1, tm1, scal1, d1) + _sigmoid(t_norm, a2, tm2, scal2, d2)


def _model_4(
    t_norm: np.ndarray,
    a1: float,
    tm1: float,
    scal1: float,
    d1: float,
    a2: float,
    tm2: float,
    scal2: float,
    d2: float,
    c: float,
    b: float,
) -> np.ndarray:
    """Model 4: Two modified sigmoids + initial fluorescence decay.

    F(T) = Sig1(T) + Sig2(T) + Id(T)

    Reference: analysis.R, s2_d_model() (lines 348-376).
    """
    return (
        _sigmoid(t_norm, a1, tm1, scal1, d1)
        + _sigmoid(t_norm, a2, tm2, scal2, d2)
        + _initial_decay(t_norm, c, b)
    )


# --- BIC computation ---


def _compute_bic(n_points: int, rss: float, n_params: int) -> float:
    """Compute Bayesian Information Criterion.

    BIC = n * ln(RSS / n) + k * ln(n)

    Reference: analysis.R line 859 (via broom::glance()).

    Args:
        n_points: Number of data points.
        rss: Residual sum of squares.
        n_params: Number of free parameters.

    Returns:
        BIC value. Lower is better.
    """
    if rss <= 0.0 or n_points <= 0:
        return np.inf
    return n_points * np.log(rss / n_points) + n_params * np.log(n_points)


# --- Starting parameter estimation ---


def _estimate_starting_parameters(
    t_norm: np.ndarray,
    f_norm: np.ndarray,
    t_range: float,
    peak_detection_config: PeakDetectionConfig | None = None,
) -> dict[str, float]:
    """Estimate starting parameters for curve fitting from derivative peak locations.

    Uses BADA's existing detect_melting_peaks() to find peak locations in the derivative
    curve, then converts them to normalized temperature starting values.

    Reference: analysis.R, make_peak_finder_nest() (lines 247-292).

    Args:
        t_norm: Normalized temperature array.
        f_norm: Normalized fluorescence array.
        t_range: Original temperature range in degrees Celsius (t_max - t_min).
            Used to scale the min_width_celsius threshold to normalized units.
        peak_detection_config: Configuration for peak detection.

    Returns:
        Dictionary with starting parameter estimates:
            "tm1": First peak location (normalized).
            "tm2": Second peak location (normalized), or tm1 + offset if only one peak.
            "c": Starting fluorescence value (f_norm[0]).
    """
    # Scale peak detection config to normalized [0, 1] domain.
    # min_width_celsius is in degrees Celsius; dividing by t_range converts
    # it to the same units as the normalized temperature axis.
    base_config = peak_detection_config or PeakDetectionConfig()
    if t_range > 0:
        normalized_width = base_config.min_width_celsius / t_range
    else:
        normalized_width = base_config.min_width_celsius
    config_for_normalized = PeakDetectionConfig(
        boundary_margin_fraction=base_config.boundary_margin_fraction,
        min_prominence_factor=base_config.min_prominence_factor,
        min_width_celsius=normalized_width,
        min_height_fraction=base_config.min_height_fraction,
        max_peaks_before_flagging=base_config.max_peaks_before_flagging,
    )

    # Fit a spline to normalized data and get derivative for peak detection
    spline, x_spline, _y_spline = get_spline(t_norm, f_norm, smoothing=0.01)
    y_derivative = get_spline_derivative(spline, x_spline)

    peak_result = detect_melting_peaks(x_spline, y_derivative, config=config_for_normalized)

    if peak_result.n_peaks_found == 0:
        # Fallback: use midpoint
        tm1 = 0.5
        tm2 = 0.5 + TM2_OFFSET_NORMALIZED
    elif peak_result.n_peaks_found == 1:
        tm1 = float(peak_result.all_peak_temperatures[0])
        tm2 = tm1 + TM2_OFFSET_NORMALIZED
    else:
        tm1 = float(peak_result.all_peak_temperatures[0])
        tm2 = float(peak_result.all_peak_temperatures[1])
        # Ensure tm1 < tm2
        if tm1 > tm2:
            tm1, tm2 = tm2, tm1

    c = float(f_norm[0]) if len(f_norm) > 0 else 0.5

    return {"tm1": tm1, "tm2": tm2, "c": max(c, MIN_INITIAL_DECAY_AMPLITUDE)}


# --- Tm extraction from isolated sigmoid ---


def _extract_tm_from_sigmoid(
    t_norm: np.ndarray,
    sigmoid_values: np.ndarray,
    t_min: float,
    t_max: float,
) -> tuple[float, float]:
    """Extract Tm from the derivative of an isolated sigmoid component.

    DSFworld extracts Tm from the derivative of each isolated sigmoid, not from the
    fitted Tm parameter directly. This is because the modified Boltzmann sigmoid
    (with decay != 0) shifts the derivative maximum away from the Tm parameter.

    Reference: analysis.R, model_tms_by_dRFU() (lines 779-808), Tm_by_dRFU_for_model()
    (lines 744-753).

    BADA adaptation: Use np.gradient() on the smooth parametric sigmoid output.
    For a sigmoid evaluated on a fine grid, numerical gradient is sufficient.

    Args:
        t_norm: Normalized temperature grid.
        sigmoid_values: Isolated sigmoid component evaluated on the grid.
        t_min: Original minimum temperature (for denormalization).
        t_max: Original maximum temperature (for denormalization).

    Returns:
        Tuple of (tm_real, max_derivative_value) where tm_real is in degrees Celsius.
    """
    # Evaluate on a fine grid for accurate derivative
    t_fine = np.linspace(float(t_norm[0]), float(t_norm[-1]), TM_EXTRACTION_GRID_POINTS)

    # Interpolate sigmoid values onto fine grid
    sig_fine = np.interp(t_fine, t_norm, sigmoid_values)

    # Numerical derivative
    dt = t_fine[1] - t_fine[0]
    derivative = np.gradient(sig_fine, dt)

    # Find maximum
    max_idx = int(np.argmax(derivative))
    tm_norm = float(t_fine[max_idx])
    max_deriv = float(derivative[max_idx])

    tm_real = _denormalize_temperature(tm_norm, t_min, t_max)

    return tm_real, max_deriv


# --- Single model fitting ---


def _fit_single_model(
    t_norm: np.ndarray,
    f_norm: np.ndarray,
    model_func: Callable[..., np.ndarray],
    p0: list[float],
    lower_bounds: list[float],
    max_iterations: int,
) -> tuple[np.ndarray | None, float | None]:
    """Fit a single parametric model to normalized data.

    Wraps scipy.optimize.curve_fit with Trust Region Reflective (TRF) method,
    which is scipy's closest equivalent to R's nlsLM() with bounds.

    Reference: DSFworld uses minpack.lm::nlsLM() in R, which is Levenberg-Marquardt
    with bounds. scipy's TRF is the closest equivalent (method='lm' does not support
    bounds in scipy).

    Args:
        t_norm: Normalized temperature array.
        f_norm: Normalized fluorescence array.
        model_func: Model function with signature f(t_norm, *params).
        p0: Starting parameter values.
        lower_bounds: Lower bounds for each parameter.
        max_iterations: Maximum number of optimizer iterations.

    Returns:
        Tuple of (fitted_params, rss) or (None, None) on convergence failure.
    """
    upper_bounds = [np.inf] * len(p0)

    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _pcov = curve_fit(
                f=model_func,
                xdata=t_norm,
                ydata=f_norm,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                method="trf",
                maxfev=max_iterations,
            )

        residuals = f_norm - model_func(t_norm, *popt)
        rss = float(np.sum(residuals**2))

        return popt, rss

    except (RuntimeError, ValueError) as e:
        logger.debug("Model fitting failed: %s", e)
        return None, None


# --- R-squared computation ---


def _compute_r_squared(f_norm: np.ndarray, f_fitted: np.ndarray) -> float:
    """Compute R-squared (coefficient of determination).

    Args:
        f_norm: Observed normalized fluorescence.
        f_fitted: Fitted (predicted) fluorescence.

    Returns:
        R-squared value.
    """
    ss_res = float(np.sum((f_norm - f_fitted) ** 2))
    ss_tot = float(np.sum((f_norm - np.mean(f_norm)) ** 2))

    if np.isclose(ss_tot, 0.0):
        return 0.0

    return 1.0 - ss_res / ss_tot


# --- Decompose model into components ---


def _decompose_model(
    t_norm: np.ndarray,
    model_name: str,
    params: np.ndarray,
) -> dict[str, np.ndarray]:
    """Decompose a fitted model into its individual components.

    Args:
        t_norm: Normalized temperature array.
        model_name: One of "model_1", "model_2", "model_3", "model_4".
        params: Fitted parameter array from curve_fit.

    Returns:
        Dictionary with keys "sigmoid_1" and optionally "sigmoid_2", "initial_decay".
    """
    components: dict[str, np.ndarray] = {}

    if model_name == "model_1":
        a, tm, scal, d = params
        components["sigmoid_1"] = _sigmoid(t_norm, a, tm, scal, d)

    elif model_name == "model_2":
        a, tm, scal, d, c, b = params
        components["sigmoid_1"] = _sigmoid(t_norm, a, tm, scal, d)
        components["initial_decay"] = _initial_decay(t_norm, c, b)

    elif model_name == "model_3":
        a1, tm1, scal1, d1, a2, tm2, scal2, d2 = params
        components["sigmoid_1"] = _sigmoid(t_norm, a1, tm1, scal1, d1)
        components["sigmoid_2"] = _sigmoid(t_norm, a2, tm2, scal2, d2)

    elif model_name == "model_4":
        a1, tm1, scal1, d1, a2, tm2, scal2, d2, c, b = params
        components["sigmoid_1"] = _sigmoid(t_norm, a1, tm1, scal1, d1)
        components["sigmoid_2"] = _sigmoid(t_norm, a2, tm2, scal2, d2)
        components["initial_decay"] = _initial_decay(t_norm, c, b)

    return components


# --- Build starting parameters and bounds for each model ---


def _build_model_params(
    model_name: str,
    starts: dict[str, float],
) -> tuple[list[float], list[float]]:
    """Build starting parameters and lower bounds for a specific model.

    Args:
        model_name: One of "model_1", "model_2", "model_3", "model_4".
        starts: Dictionary from _estimate_starting_parameters().

    Returns:
        Tuple of (p0, lower_bounds).
    """
    tm1 = starts["tm1"]
    tm2 = starts["tm2"]
    c = starts["c"]

    if model_name == "model_1":
        p0 = [DEFAULT_AMPLITUDE, tm1, DEFAULT_SCALE, DEFAULT_DECAY]
        lower = [MIN_AMPLITUDE, MIN_TM, MIN_SCALE, MIN_DECAY]

    elif model_name == "model_2":
        p0 = [DEFAULT_AMPLITUDE, tm1, DEFAULT_SCALE, DEFAULT_DECAY, c, DEFAULT_DECAY_RATE]
        lower = [
            MIN_AMPLITUDE,
            MIN_TM,
            MIN_SCALE,
            MIN_DECAY,
            MIN_INITIAL_DECAY_AMPLITUDE,
            MIN_INITIAL_DECAY_RATE,
        ]

    elif model_name == "model_3":
        p0 = [
            DEFAULT_AMPLITUDE,
            tm1,
            DEFAULT_SCALE,
            DEFAULT_DECAY,
            DEFAULT_AMPLITUDE_2,
            tm2,
            DEFAULT_SCALE,
            DEFAULT_DECAY_2,
        ]
        lower = [
            MIN_AMPLITUDE_TWO_SIGMOID,
            MIN_TM,
            MIN_SCALE,
            MIN_DECAY,
            MIN_AMPLITUDE_TWO_SIGMOID,
            MIN_TM,
            MIN_SCALE,
            MIN_DECAY,
        ]

    elif model_name == "model_4":
        p0 = [
            DEFAULT_AMPLITUDE,
            tm1,
            DEFAULT_SCALE,
            DEFAULT_DECAY,
            DEFAULT_AMPLITUDE_2,
            tm2,
            DEFAULT_SCALE,
            DEFAULT_DECAY_2,
            c,
            DEFAULT_DECAY_RATE,
        ]
        lower = [
            MIN_AMPLITUDE_TWO_SIGMOID,
            MIN_TM,
            MIN_SCALE,
            MIN_DECAY,
            MIN_AMPLITUDE_TWO_SIGMOID,
            MIN_TM,
            MIN_SCALE,
            MIN_DECAY,
            MIN_INITIAL_DECAY_AMPLITUDE,
            MIN_INITIAL_DECAY_RATE,
        ]

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return p0, lower


# --- Parameter name mapping ---

MODEL_PARAM_NAMES: dict[str, list[str]] = {
    "model_1": ["A", "Tm", "scal", "d"],
    "model_2": ["A", "Tm", "scal", "d", "C", "b"],
    "model_3": ["A1", "Tm1", "scal1", "d1", "A2", "Tm2", "scal2", "d2"],
    "model_4": ["A1", "Tm1", "scal1", "d1", "A2", "Tm2", "scal2", "d2", "C", "b"],
}

MODEL_N_PARAMS: dict[str, int] = {
    "model_1": 4,
    "model_2": 6,
    "model_3": 8,
    "model_4": 10,
}

MODEL_FUNCTIONS = {
    "model_1": _model_1,
    "model_2": _model_2,
    "model_3": _model_3,
    "model_4": _model_4,
}

ALL_MODEL_NAMES = ["model_1", "model_2", "model_3", "model_4"]

# Mapping from two-sigmoid models to their single-sigmoid counterparts.
# Used when the minimum Tm separation check rejects a two-sigmoid solution.
_TWO_TO_SINGLE_SIGMOID_FALLBACK: dict[str, str] = {
    "model_3": "model_1",  # two sigmoids → single sigmoid
    "model_4": "model_2",  # two sigmoids + decay → single sigmoid + decay
}


def _extract_tms_from_components(
    t_norm: np.ndarray,
    components: dict[str, np.ndarray],
    t_min: float,
    t_max: float,
) -> tuple[float, float, float | None]:
    """Extract primary and secondary Tm from decomposed sigmoid components.

    When two sigmoids exist, the one with the larger derivative maximum is
    selected as the primary Tm. The optimizer may swap sigmoid_1/sigmoid_2
    labels arbitrarily, so selection by derivative magnitude is required.

    Args:
        t_norm: Normalized temperature array.
        components: Decomposed model components from _decompose_model().
        t_min: Original minimum temperature (for denormalization).
        t_max: Original maximum temperature (for denormalization).

    Returns:
        Tuple of (tm_primary, max_derivative_value, tm_secondary).
        tm_secondary is None when only one sigmoid is present.
    """
    tm1_real, deriv1 = _extract_tm_from_sigmoid(t_norm, components["sigmoid_1"], t_min, t_max)

    if "sigmoid_2" not in components:
        return tm1_real, deriv1, None

    tm2_real, deriv2 = _extract_tm_from_sigmoid(t_norm, components["sigmoid_2"], t_min, t_max)

    # Primary Tm is from the sigmoid with the larger derivative maximum
    if deriv2 > deriv1:
        return tm2_real, deriv2, tm1_real
    return tm1_real, deriv1, tm2_real


# --- Main public entry point ---


def fit_dsf_models(
    temperature: np.ndarray,
    fluorescence: np.ndarray,
    tm_method: TmMethod | str = TmMethod.AUTO,
    config: ModelFittingConfig | None = None,
    peak_detection_config: PeakDetectionConfig | None = None,
) -> ModelFitResult:
    """Fit DSFworld parametric models to a melting curve and extract Tm.

    This is the main public entry point for multi-model curve fitting. It normalizes
    the data, estimates starting parameters, fits models, selects the best by BIC,
    decomposes into components, and extracts Tm from the derivative of each isolated
    sigmoid.

    Args:
        temperature: Temperature array in degrees Celsius.
        fluorescence: Fluorescence array (RFU).
        tm_method: Which model(s) to fit. TmMethod.AUTO fits all 4 and selects by BIC.
            A specific model (e.g., TmMethod.MODEL_2) fits only that model.
        config: Fitting configuration. Uses defaults if None.
        peak_detection_config: Configuration for peak detection (used for starting
            parameter estimation and derivative fallback).

    Returns:
        ModelFitResult with Tm, model selection, and quality information.
    """
    if config is None:
        config = ModelFittingConfig()

    if isinstance(tm_method, str):
        tm_method = TmMethod(tm_method)

    # Step 1: Normalize data
    temp_arr = np.asarray(temperature, dtype=float)
    fluor_arr = np.asarray(fluorescence, dtype=float)
    t_norm, f_norm, t_min, t_max, f_min, f_max = _normalize_data(temp_arr, fluor_arr)

    # Early exit: flat fluorescence has no transition to fit
    if np.isclose(f_max, f_min):
        return _handle_all_models_failed(temp_arr, fluor_arr, config, peak_detection_config, {})

    # Step 2: Estimate starting parameters
    t_range = t_max - t_min
    starts = _estimate_starting_parameters(t_norm, f_norm, t_range, peak_detection_config)

    # Step 3: Determine which models to fit
    if tm_method == TmMethod.AUTO:
        models_to_fit = ALL_MODEL_NAMES
    else:
        models_to_fit = [tm_method.value]

    # Step 4: Fit models
    fit_results: dict[str, tuple[np.ndarray, float]] = {}

    for model_name in models_to_fit:
        model_func = MODEL_FUNCTIONS[model_name]
        p0, lower_bounds = _build_model_params(model_name, starts)
        popt, rss = _fit_single_model(
            t_norm, f_norm, model_func, p0, lower_bounds, config.max_iterations
        )
        if popt is not None and rss is not None:
            fit_results[model_name] = (popt, rss)

    # Step 5: Compute BIC and select best model
    n_points = len(t_norm)
    bic_values: dict[str, float] = {}

    for model_name, (popt, rss) in fit_results.items():
        n_params = MODEL_N_PARAMS[model_name]
        bic_values[model_name] = _compute_bic(n_points, rss, n_params)

    if not bic_values:
        # All models failed
        return _handle_all_models_failed(
            temperature, fluorescence, config, peak_detection_config, bic_values
        )

    selected_model = min(bic_values, key=bic_values.get)  # ty:ignore[no-matching-overload]
    selected_params, _ = fit_results[selected_model]

    # Step 6: Decompose into components
    components = _decompose_model(t_norm, selected_model, selected_params)

    # Step 7: Extract Tm from isolated sigmoid(s)
    tm_real, max_deriv, tm_secondary = _extract_tms_from_components(
        t_norm, components, t_min, t_max
    )

    # Step 7b: Reject two-sigmoid models with insufficient Tm separation.
    # When the two extracted Tms are closer than MIN_TM_SEPARATION_CELSIUS,
    # the second sigmoid is likely absorbing post-transition behavior rather
    # than representing a genuine second unfolding transition. Fall back to
    # the corresponding single-sigmoid model if it converged.
    if (
        tm_secondary is not None
        and abs(tm_real - tm_secondary) < MIN_TM_SEPARATION_CELSIUS
        and selected_model in _TWO_TO_SINGLE_SIGMOID_FALLBACK
    ):
        fallback_model = _TWO_TO_SINGLE_SIGMOID_FALLBACK[selected_model]
        if fallback_model in fit_results:
            selected_model = fallback_model
            selected_params, _ = fit_results[selected_model]
            components = _decompose_model(t_norm, selected_model, selected_params)
            tm_real, max_deriv, tm_secondary = _extract_tms_from_components(
                t_norm, components, t_min, t_max
            )

    # Step 8: Compute R-squared
    model_func = MODEL_FUNCTIONS[selected_model]
    f_fitted = model_func(t_norm, *selected_params)
    r_squared = _compute_r_squared(f_norm, f_fitted)

    # Build parameter name mapping
    param_names = MODEL_PARAM_NAMES[selected_model]
    model_parameters: dict[str, float] = {
        name: float(val) for name, val in zip(param_names, selected_params)
    }

    return ModelFitResult(
        tm=tm_real,
        tm_secondary=tm_secondary,
        max_derivative_value=max_deriv,
        selected_model=selected_model,
        bic_values=bic_values,
        model_parameters=model_parameters,
        r_squared=r_squared,
        component_curves=components,
        is_valid=True,
        quality_flags=[],
    )


def _handle_all_models_failed(
    temperature: np.ndarray,
    fluorescence: np.ndarray,
    config: ModelFittingConfig,
    peak_detection_config: PeakDetectionConfig | None,
    bic_values: dict[str, float],
) -> ModelFitResult:
    """Handle the case when all parametric models fail to converge.

    If fallback_to_derivative is enabled, uses the existing BADA derivative method.
    Otherwise, returns a failed result with NaN values.

    Args:
        temperature: Raw temperature array.
        fluorescence: Raw fluorescence array.
        config: Fitting configuration.
        peak_detection_config: Peak detection configuration for fallback.
        bic_values: (Empty) BIC values dict.

    Returns:
        ModelFitResult with derivative fallback or NaN values.
    """
    quality_flags = ["all_models_failed"]

    if config.fallback_to_derivative:
        quality_flags.append("fallback_to_derivative")

        spline, x_spline, _y_spline = get_spline(temperature, fluorescence, smoothing=0.01)
        y_derivative = get_spline_derivative(spline, x_spline)
        peak_result = detect_melting_peaks(x_spline, y_derivative, config=peak_detection_config)

        return ModelFitResult(
            tm=peak_result.tm,
            tm_secondary=None,
            max_derivative_value=peak_result.max_derivative_value,
            selected_model="derivative",
            bic_values=bic_values,
            model_parameters={},
            r_squared=np.nan,
            component_curves={},
            is_valid=peak_result.is_valid,
            quality_flags=quality_flags,
        )

    return ModelFitResult(
        tm=np.nan,
        tm_secondary=None,
        max_derivative_value=np.nan,
        selected_model="none",
        bic_values=bic_values,
        model_parameters={},
        r_squared=np.nan,
        component_curves={},
        is_valid=False,
        quality_flags=quality_flags,
    )


def evaluate_fit_result(
    result: ModelFitResult,
    temperature: np.ndarray,
    fluorescence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Evaluate a fitted model at given temperatures, returning real-scale values.

    Re-normalizes the input data (identically to how it was done during fitting),
    evaluates the selected model and its decomposed components on the x_eval grid,
    then denormalizes fluorescence values back to the original scale.

    Args:
        result: A valid ModelFitResult from fit_dsf_models().
        temperature: Temperature array in degrees Celsius (same data used for fitting).
        fluorescence: Fluorescence array in RFU (same data used for fitting).

    Returns:
        Tuple of (x_eval, y_fitted, component_curves) where:
            x_eval: Temperature grid in degrees Celsius.
            y_fitted: Full model prediction in original fluorescence units.
            component_curves: Dict mapping component name to fluorescence array
                (in original units, as additive contributions without baseline offset).
    """
    if result.selected_model not in MODEL_FUNCTIONS:
        return np.array([]), np.array([]), {}

    t_norm, _f_norm, _t_min, _t_max, f_min, f_max = _normalize_data(temperature, fluorescence)
    f_range = f_max - f_min

    # Reconstruct parameter array in correct order
    param_names = MODEL_PARAM_NAMES[result.selected_model]
    params = np.array([result.model_parameters[name] for name in param_names])

    # Evaluate full model
    model_func = MODEL_FUNCTIONS[result.selected_model]
    y_fitted_norm = model_func(t_norm, *params)

    # Denormalize: y_real = y_norm * f_range + f_min
    x_eval = temperature.copy()
    y_fitted = y_fitted_norm * f_range + f_min

    # Decompose and scale components (additive contributions)
    components_norm = _decompose_model(t_norm, result.selected_model, params)
    component_curves: dict[str, np.ndarray] = {
        name: values * f_range for name, values in components_norm.items()
    }

    return x_eval, y_fitted, component_curves
