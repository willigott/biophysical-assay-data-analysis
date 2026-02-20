"""Robust peak detection for DSF melting temperature determination.

Replaces naive np.argmax() with scipy.signal.find_peaks() and validated
peak filtering. The algorithm processes a derivative curve through five stages:

1. Signal Characterization — compute derivative range, temp resolution, flat detection
2. Noise Estimation — MAD-based robust noise estimate
3. Peak Detection — scipy.signal.find_peaks with height/width constraints
4. Peak Filtering — boundary exclusion and prominence filtering
5. Result Assembly — confidence scoring and result construction

References:
    [4] Sun et al. 2020, Protein Science 29(1), 19-27. doi:10.1002/pro.3703
    [9] Wu et al. 2023, STAR Protocols 4(4), 102688. doi:10.1016/j.xpro.2023.102688
    [10] Niesen et al. 2007, Nature Protocols 2(9), 2212-2221. doi:10.1038/nprot.2007.321
    [11] Hampel 1974, JASA 69(346), 383-393. doi:10.1080/01621459.1974.10482962
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_prominences

# Derivative ranges below this threshold indicate a flat curve (no unfolding transition).
# A flat derivative means constant fluorescence, which occurs with empty wells,
# pre-denatured protein, or instrument failures. See SimpleDSFviewer [4].
FLAT_DERIVATIVE_EPSILON = 1e-10

# Floor for noise estimate to avoid division-by-zero in confidence computation.
# Perfectly smooth synthetic curves have zero noise; this prevents NaN/Inf downstream.
NOISE_FLOOR = 1e-12

# Scaling factor to convert MAD to a consistent estimator of sigma for Gaussian noise.
# For a normal distribution, MAD = sigma * Phi^{-1}(3/4) ≈ sigma * 0.6745,
# so sigma ≈ MAD / 0.6745 ≈ MAD * 1.4826. See Hampel 1974 [11].
MAD_SCALE_FACTOR = 1.4826

# --- Confidence score weights and thresholds (Stage 5) ---
# A peak with prominence >= SNR_THRESHOLD * noise_sigma gets full SNR credit.
# 5-sigma is the standard signal detection threshold (used in physics, e.g., Higgs boson
# discovery). In DSF, cooperative unfolding typically has SNR >> 5; questionable curves
# have SNR 2-5; pure noise has SNR < 1.
SNR_WEIGHT = 0.50
SNR_THRESHOLD = 5.0

# A peak accounting for >= PROMINENCE_THRESHOLD of the total derivative range indicates
# a clean, cooperative unfolding transition. This is the continuous-valued generalization
# of SimpleDSFviewer's binary 10% threshold [4].
PROMINENCE_WEIGHT = 0.35
PROMINENCE_THRESHOLD = 0.50

# Position centrality penalizes edge-adjacent peaks. ~60% of proteins have Tm between
# 45-70°C [9], centered in the typical 25-95°C scan range.
POSITION_WEIGHT = 0.15


@dataclass(frozen=True)
class PeakDetectionConfig:
    """Configuration for peak detection with adaptive thresholds.

    All thresholds are relative to signal properties, making them adaptive
    to different instruments and protein systems.

    Attributes:
        boundary_margin_fraction: Fraction of temperature range to exclude at each edge
            (~2°C on a 25-95°C scan). DSF instruments show fluorescence drift at ramp
            start and dye degradation near upper limit [9].
        min_prominence_factor: Min peak prominence as fraction of derivative range.
            SimpleDSFviewer requires peaks to exceed 10% of the highest peak's
            height/width to be recognized as a real transition [4].
        min_width_celsius: Min peak width at half-prominence, in °C. Real unfolding
            transitions span 5-15°C in the first derivative [10]. Multi-domain proteins
            have narrower transitions (~3-5°C). Noise spikes are <1°C.
        min_height_fraction: Min peak height as fraction of derivative range. Pre-filter
            to eliminate "peaks" in flat derivatives (denatured/empty wells).
        max_peaks_before_flagging: Flag curve as problematic if this many peaks are found.
            SimpleDSFviewer flags wells with >=4 peaks as denatured or empty [4].
    """

    boundary_margin_fraction: float = 0.03
    min_prominence_factor: float = 0.10
    min_width_celsius: float = 2.0
    min_height_fraction: float = 0.05
    max_peaks_before_flagging: int = 4


@dataclass(frozen=True)
class PeakDetectionResult:
    """Result of peak detection on a derivative curve.

    Attributes:
        tm: Melting temperature at the most prominent valid peak. np.nan if no valid
            peak found.
        max_derivative_value: Derivative value at Tm. np.nan if no valid peak.
        peak_confidence: Confidence score in [0.0, 1.0]. 0.0 means no peak found.
        n_peaks_found: Number of valid peaks detected after filtering.
        all_peak_temperatures: All valid peak temperatures, sorted by prominence
            (most prominent first).
        all_peak_prominences: Corresponding prominences for each peak.
        is_valid: Whether a valid peak was found and the curve is not flagged
            as problematic.
        quality_flags: Quality issue descriptions (e.g., "no_valid_peak",
            "flat_derivative", "too_many_peaks").
    """

    tm: float
    max_derivative_value: float
    peak_confidence: float
    n_peaks_found: int
    all_peak_temperatures: np.ndarray
    all_peak_prominences: np.ndarray
    is_valid: bool
    quality_flags: list[str]


@dataclass(frozen=True)
class _SignalCharacteristics:
    """Internal result from Stage 1: Signal Characterization.

    Attributes:
        derivative_range: max(y_derivative) - min(y_derivative). Total dynamic range
            of the derivative signal; used as normalization denominator for relative
            thresholds.
        temp_resolution: (x_spline[-1] - x_spline[0]) / (len(x_spline) - 1). Degrees
            Celsius per sample point. Used to convert width thresholds from °C to samples.
        is_flat: True if derivative_range < FLAT_DERIVATIVE_EPSILON, indicating no
            meaningful transitions in the curve.
    """

    derivative_range: float
    temp_resolution: float
    is_flat: bool


def _characterize_signal(x_spline: np.ndarray, y_derivative: np.ndarray) -> _SignalCharacteristics:
    """Stage 1: Characterize the derivative signal.

    Computes the derivative range (total dynamic range) and temperature resolution
    (°C per sample), and detects flat derivatives that indicate no unfolding transition.

    A flat derivative (derivative_range < FLAT_DERIVATIVE_EPSILON) means constant
    fluorescence, which occurs with empty wells, pre-denatured protein, or instrument
    failures. SimpleDSFviewer uses a similar check [4].

    Args:
        x_spline: Temperature array (monotonically increasing).
        y_derivative: First derivative of fluorescence (dF/dT), same length as x_spline.

    Returns:
        _SignalCharacteristics with derivative_range, temp_resolution, and is_flat flag.

    Reference:
        Sun et al. 2020 [4] — SimpleDSFviewer classifies flat traces as "no transition".
        Niesen et al. 2007 [10] — flat traces described as a common DSF failure mode.
    """
    derivative_range = float(np.max(y_derivative) - np.min(y_derivative))
    temp_resolution = float((x_spline[-1] - x_spline[0]) / (len(x_spline) - 1))
    is_flat = derivative_range < FLAT_DERIVATIVE_EPSILON

    return _SignalCharacteristics(
        derivative_range=derivative_range,
        temp_resolution=temp_resolution,
        is_flat=is_flat,
    )


def _estimate_derivative_noise(y_derivative: np.ndarray) -> float:
    """Stage 2: Estimate noise level in the derivative curve using MAD.

    Uses the Median Absolute Deviation (MAD) of point-to-point differences in the
    derivative. This is robust to outliers: up to 50% of the data can belong to real
    peaks without inflating the estimate, unlike standard deviation.

    The point-to-point differences capture local variability (noise) while being
    insensitive to the global shape (peaks). The MAD of these differences is then
    scaled by 1.4826 to convert to a Gaussian-equivalent standard deviation.

    The result is floored at NOISE_FLOOR (1e-12) to avoid division-by-zero in
    confidence computation when the curve is perfectly smooth (common in synthetic
    test data).

    Args:
        y_derivative: First derivative of fluorescence (dF/dT).

    Returns:
        Estimated noise standard deviation (sigma), floored at NOISE_FLOOR.

    Reference:
        Hampel 1974 [11] — MAD as robust location estimator with 50% breakdown point.
            The 1.4826 scaling factor makes MAD a consistent estimator of sigma for
            normally distributed data.
    """
    diffs = np.diff(y_derivative)
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    noise_sigma = MAD_SCALE_FACTOR * mad

    return max(noise_sigma, NOISE_FLOOR)


def _detect_raw_peaks(
    y_derivative: np.ndarray,
    signal: _SignalCharacteristics,
    config: PeakDetectionConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Stage 3: Detect raw peaks using scipy.signal.find_peaks.

    Calls find_peaks with height and width constraints derived from signal
    characteristics and configuration. Prominence filtering is intentionally
    deferred to Stage 4 (after boundary exclusion), so that edge peaks are
    removed before prominence comparison.

    The height threshold is an absolute minimum: peaks below
    min(y_derivative) + min_height_fraction * derivative_range are not
    considered. The width threshold filters out noise spikes narrower than
    min_width_celsius (measured at half-prominence, i.e., FWHM).

    Args:
        y_derivative: First derivative of fluorescence (dF/dT).
        signal: Signal characteristics from Stage 1.
        config: Peak detection configuration.

    Returns:
        Tuple of (peak_indices, properties_dict) from scipy.signal.find_peaks.
        properties_dict contains "peak_heights", "widths", "width_heights",
        "left_ips", and "right_ips".

    Reference:
        scipy.signal.find_peaks:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        Niesen et al. 2007 [10] — real unfolding transitions span 5-15°C in the
            first derivative; multi-domain proteins have narrower transitions (~3-5°C).
    """
    min_width_samples = config.min_width_celsius / signal.temp_resolution
    min_height = float(np.min(y_derivative)) + config.min_height_fraction * signal.derivative_range

    peak_indices, properties = find_peaks(
        y_derivative,
        height=min_height,
        width=min_width_samples,
        rel_height=0.5,
    )

    return peak_indices, properties


@dataclass(frozen=True)
class _FilteredPeaks:
    """Internal result from Stage 4: Peak Filtering.

    Attributes:
        peak_indices: Indices into the derivative array for surviving peaks,
            sorted by prominence (most prominent first).
        prominences: Prominence values corresponding to each peak, sorted
            descending (most prominent first).
        quality_flags: Quality issues detected during filtering (e.g.,
            "too_many_peaks").
    """

    peak_indices: np.ndarray
    prominences: np.ndarray
    quality_flags: list[str]


def _filter_peaks(
    peak_indices: np.ndarray,
    x_spline: np.ndarray,
    y_derivative: np.ndarray,
    signal: _SignalCharacteristics,
    config: PeakDetectionConfig,
) -> _FilteredPeaks:
    """Stage 4: Filter peaks by boundary exclusion and prominence.

    Applies two sequential filters to the raw peaks from Stage 3:

    1. **Boundary exclusion:** Peaks within boundary_margin_fraction of the
       temperature range edges are removed. DSF instruments show fluorescence
       drift at ramp start and dye degradation near the upper limit [9].

    2. **Prominence filtering:** Peaks with prominence below
       min_prominence_factor * derivative_range are removed. Prominence measures
       how much a peak stands out from the surrounding signal — it is more
       discriminating than raw height because it accounts for the local baseline.
       This implements the SimpleDSFviewer 10% criterion [4].

    After filtering, peaks are sorted by prominence (descending) so the most
    prominent peak is first. If the number of surviving peaks exceeds
    max_peaks_before_flagging, a "too_many_peaks" quality flag is set.

    Args:
        peak_indices: Peak indices from Stage 3 (_detect_raw_peaks).
        x_spline: Temperature array (monotonically increasing).
        y_derivative: First derivative of fluorescence (dF/dT).
        signal: Signal characteristics from Stage 1.
        config: Peak detection configuration.

    Returns:
        _FilteredPeaks with surviving peak indices (sorted by prominence),
        their prominences, and quality flags.

    Reference:
        Sun et al. 2020 [4] — 10% prominence threshold; >=4 peaks = denatured.
        Wu et al. 2023 [9] — recommends excluding scan endpoints from analysis.
        scipy.signal.peak_prominences:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html
    """
    quality_flags: list[str] = []

    if peak_indices.size == 0:
        return _FilteredPeaks(
            peak_indices=np.array([], dtype=int),
            prominences=np.array([]),
            quality_flags=["no_valid_peak"],
        )

    # --- Boundary exclusion ---
    temp_range = float(x_spline[-1] - x_spline[0])
    margin = config.boundary_margin_fraction * temp_range
    lower_bound = float(x_spline[0]) + margin
    upper_bound = float(x_spline[-1]) - margin

    boundary_mask = np.array(
        [lower_bound <= float(x_spline[idx]) <= upper_bound for idx in peak_indices]
    )
    surviving_indices = peak_indices[boundary_mask]

    if surviving_indices.size == 0:
        return _FilteredPeaks(
            peak_indices=np.array([], dtype=int),
            prominences=np.array([]),
            quality_flags=["no_valid_peak"],
        )

    # --- Prominence computation and filtering ---
    prominences_array, _, _ = peak_prominences(y_derivative, surviving_indices)
    min_prominence = config.min_prominence_factor * signal.derivative_range

    prominence_mask = prominences_array >= min_prominence
    surviving_indices = surviving_indices[prominence_mask]
    prominences_array = prominences_array[prominence_mask]

    if surviving_indices.size == 0:
        return _FilteredPeaks(
            peak_indices=np.array([], dtype=int),
            prominences=np.array([]),
            quality_flags=["no_valid_peak"],
        )

    # --- Too many peaks check ---
    if len(surviving_indices) >= config.max_peaks_before_flagging:
        quality_flags.append("too_many_peaks")

    # --- Sort by prominence (descending) ---
    sort_order = np.argsort(prominences_array)[::-1]
    surviving_indices = surviving_indices[sort_order]
    prominences_array = prominences_array[sort_order]

    return _FilteredPeaks(
        peak_indices=surviving_indices,
        prominences=prominences_array,
        quality_flags=quality_flags,
    )


def _compute_peak_confidence(
    prominence: float,
    noise_sigma: float,
    derivative_range: float,
    tm: float,
    x_spline: np.ndarray,
) -> float:
    """Stage 5: Compute a confidence score for the detected Tm.

    Combines three independent indicators into a single [0.0, 1.0] value.
    Each factor is individually clamped to [0.0, 1.0] before weighting.

    Factor 1 — Signal-to-Noise Ratio (weight: 0.50):
        A peak whose prominence is 5x the noise level gets a perfect score.
        The 5-sigma threshold is standard in signal detection. In DSF, a
        well-defined cooperative unfolding transition has SNR >> 5; questionable
        curves have SNR 2-5; pure noise has SNR < 1.

    Factor 2 — Relative Prominence (weight: 0.35):
        A peak accounting for >=50% of the total derivative range indicates a
        clean, cooperative unfolding with a single dominant transition. This is
        the continuous-valued generalization of SimpleDSFviewer's binary 10%
        threshold [4].

    Factor 3 — Position Centrality (weight: 0.15):
        Score is 1.0 at center of scan range, 0.0 at edges. Penalizes peaks
        near scan boundaries even after boundary exclusion, because edge-adjacent
        peaks are more likely to be artifacts. ~60% of proteins have Tm between
        45-70°C [9].

    Args:
        prominence: Prominence of the peak (from peak_prominences).
        noise_sigma: Estimated noise standard deviation from Stage 2.
        derivative_range: Total dynamic range of the derivative (max - min).
        tm: Detected melting temperature.
        x_spline: Temperature array (for computing position centrality).

    Returns:
        Confidence score in [0.0, 1.0].

    Reference:
        Sun et al. 2020 [4] — 10% prominence threshold.
        Wu et al. 2023 [9] — ~60% of proteins have Tm between 45-70°C.
        Kotov et al. 2021 [3] — BS-factor quality metric (analogous concept).
        Lee et al. 2019 [2] — TSA-CRAFT R² > 0.98 criterion (analogous concept).
    """
    # Factor 1: Signal-to-Noise Ratio
    snr_score = min(1.0, prominence / (SNR_THRESHOLD * noise_sigma))
    snr_score = max(0.0, snr_score)

    # Factor 2: Relative Prominence
    prom_score = min(1.0, prominence / (PROMINENCE_THRESHOLD * derivative_range))
    prom_score = max(0.0, prom_score)

    # Factor 3: Position Centrality
    temp_range = float(x_spline[-1] - x_spline[0])
    normalized_position = (tm - float(x_spline[0])) / temp_range
    position_score = 1.0 - 2.0 * abs(normalized_position - 0.5)
    position_score = max(0.0, min(1.0, position_score))

    confidence = (
        SNR_WEIGHT * snr_score + PROMINENCE_WEIGHT * prom_score + POSITION_WEIGHT * position_score
    )
    return max(0.0, min(1.0, confidence))


def _assemble_result(
    filtered: _FilteredPeaks,
    x_spline: np.ndarray,
    y_derivative: np.ndarray,
    noise_sigma: float,
    derivative_range: float,
) -> PeakDetectionResult:
    """Stage 5: Assemble the final PeakDetectionResult from filtered peaks.

    If no peaks survive filtering, returns a result with tm=NaN, confidence=0.0,
    and is_valid=False. Otherwise, selects the most prominent peak as Tm,
    computes the confidence score, and assembles all result fields.

    Args:
        filtered: Filtered peaks from Stage 4.
        x_spline: Temperature array.
        y_derivative: First derivative of fluorescence.
        noise_sigma: Estimated noise standard deviation from Stage 2.
        derivative_range: Total dynamic range of the derivative from Stage 1.

    Returns:
        PeakDetectionResult with Tm, confidence, and quality information.
    """
    if filtered.peak_indices.size == 0:
        return PeakDetectionResult(
            tm=np.nan,
            max_derivative_value=np.nan,
            peak_confidence=0.0,
            n_peaks_found=0,
            all_peak_temperatures=np.array([]),
            all_peak_prominences=np.array([]),
            is_valid=False,
            quality_flags=filtered.quality_flags,
        )

    # Peaks are already sorted by prominence (most prominent first) from Stage 4.
    best_peak_idx = filtered.peak_indices[0]
    tm = float(x_spline[best_peak_idx])
    max_derivative_value = float(y_derivative[best_peak_idx])
    best_prominence = float(filtered.prominences[0])

    all_peak_temps = np.array([float(x_spline[idx]) for idx in filtered.peak_indices])
    is_valid = "too_many_peaks" not in filtered.quality_flags

    confidence = _compute_peak_confidence(
        prominence=best_prominence,
        noise_sigma=noise_sigma,
        derivative_range=derivative_range,
        tm=tm,
        x_spline=x_spline,
    )

    return PeakDetectionResult(
        tm=tm,
        max_derivative_value=max_derivative_value,
        peak_confidence=confidence,
        n_peaks_found=len(filtered.peak_indices),
        all_peak_temperatures=all_peak_temps,
        all_peak_prominences=filtered.prominences,
        is_valid=is_valid,
        quality_flags=filtered.quality_flags,
    )


def _build_flat_result() -> PeakDetectionResult:
    """Build a PeakDetectionResult for a flat derivative curve."""
    return PeakDetectionResult(
        tm=np.nan,
        max_derivative_value=np.nan,
        peak_confidence=0.0,
        n_peaks_found=0,
        all_peak_temperatures=np.array([]),
        all_peak_prominences=np.array([]),
        is_valid=False,
        quality_flags=["flat_derivative"],
    )


def detect_melting_peaks(
    x_spline: np.ndarray,
    y_derivative: np.ndarray,
    config: PeakDetectionConfig | None = None,
) -> PeakDetectionResult:
    """Detect melting temperature peaks in a DSF derivative curve.

    Processes the derivative curve through a multi-stage pipeline:
    1. Signal Characterization — derivative range, temp resolution, flat detection
    2. Noise Estimation — MAD-based robust noise estimate
    3. Peak Detection — scipy.signal.find_peaks with height/width constraints
    4. Peak Filtering — boundary exclusion and prominence-based filtering
    5. Result Assembly — confidence scoring and result construction

    The most prominent peak (after boundary exclusion and prominence filtering)
    is selected as Tm. The confidence score combines signal-to-noise ratio (50%),
    relative prominence (35%), and position centrality (15%).

    Args:
        x_spline: Temperature array (monotonically increasing, typically 1000 points).
        y_derivative: First derivative of fluorescence (dF/dT), same length as x_spline.
        config: Peak detection parameters. Uses defaults if None.

    Returns:
        PeakDetectionResult with detected Tm, confidence score, and quality flags.
    """
    if config is None:
        config = PeakDetectionConfig()

    # Stage 1: Signal Characterization
    signal = _characterize_signal(x_spline, y_derivative)

    if signal.is_flat:
        return _build_flat_result()

    # Stage 2: Noise Estimation
    noise_sigma = _estimate_derivative_noise(y_derivative)

    # Stage 3: Peak Detection
    peak_indices, _properties = _detect_raw_peaks(y_derivative, signal, config)

    if peak_indices.size == 0:
        return PeakDetectionResult(
            tm=np.nan,
            max_derivative_value=np.nan,
            peak_confidence=0.0,
            n_peaks_found=0,
            all_peak_temperatures=np.array([]),
            all_peak_prominences=np.array([]),
            is_valid=False,
            quality_flags=["no_valid_peak"],
        )

    # Stage 4: Peak Filtering (boundary exclusion + prominence filtering)
    filtered = _filter_peaks(peak_indices, x_spline, y_derivative, signal, config)

    # Stage 5: Result Assembly (confidence scoring + result construction)
    return _assemble_result(filtered, x_spline, y_derivative, noise_sigma, signal.derivative_range)
