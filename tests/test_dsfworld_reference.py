"""Tests against 5 DSFbase reference curves with known melting temperatures.

Validates that BADA's multi-model curve fitting produces Tm values consistent
with the DSFworld reference implementation (Wu et al. 2024 [1]) using 5 curves
from the DSFbase benchmark dataset [2].

The 5 curves cover two model types that DSFworld's BIC selects in practice:
    - 3 model_2 curves (sigmoid + initial decay) at low, mid, and high Tm
    - 2 model_4 curves (two sigmoids + initial decay) with two transitions

Each curve is validated against the Tm values from Data_S3_Tmas.csv in the
DSFbase repository, with a tolerance of 1.0 degrees C. This tolerance
accounts for (a) different nonlinear optimizers — BADA uses scipy's TRF
while DSFworld uses R's nlsLM, leading to different BIC model selections
for some curves, and (b) cross-platform numerical variation in curve_fit
convergence across BLAS/LAPACK backends. A tolerance of 1.0 degrees C is
well within DSF experimental reproducibility (0.5-1.0 degrees C between
replicates).

References:
    [1] Wu et al. 2024, Protein Science 33(6), e5022. doi:10.1002/pro.5022
    [2] DSFbase v001: https://github.com/taiawu/dsfbase
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bada.processing.model_fitting import TmMethod, fit_dsf_models

REFERENCE_DATA_PATH = Path(__file__).parent / "data" / "dsfworld_reference_curves.json"

TM_TOLERANCE = 1.0  # degrees Celsius


@pytest.fixture(scope="module")
def reference_data() -> dict[str, Any]:
    """Load the 5 DSFbase reference curves from JSON."""
    with open(REFERENCE_DATA_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def reference_temperature(reference_data: dict[str, Any]) -> np.ndarray:
    """Shared temperature array (25-94 C, 70 points)."""
    return np.array(reference_data["temperature"])


@pytest.fixture(scope="module")
def reference_results(
    reference_data: dict[str, Any], reference_temperature: np.ndarray
) -> list[dict[str, Any]]:
    """Fit all 5 reference curves once and cache the results.

    Each entry contains the original curve metadata plus the BADA fit result.
    """
    results = []
    for curve in reference_data["curves"]:
        fluorescence = np.array(curve["fluorescence"])
        result = fit_dsf_models(reference_temperature, fluorescence, tm_method=TmMethod.AUTO)
        results.append({**curve, "result": result})
    return results


class TestDSFworldReferenceTmAccuracy:
    """Validate Tm accuracy against DSFbase ground truth within 1.0 C.

    Each of the 5 curves was selected from DSFbase v001 to represent common
    DSF curve types. The tolerance of 1.0 C accounts for BADA's different
    optimizer (scipy TRF vs R's nlsLM) and cross-platform convergence
    differences, while remaining within DSF experimental reproducibility.

    Reference: Wu et al. 2024 [1], Data_S3_Tmas.csv in DSFbase [2].
    """

    @pytest.mark.parametrize(
        "curve_index, curve_id",
        [
            (0, "DSFbase01_canon_ID000020"),
            (1, "DSFbase01_canon_ID000207"),
            (2, "DSFbase01_canon_ID000012"),
            (3, "DSFbase01_noncanon_ID004714"),
            (4, "DSFbase01_noncanon_ID004711"),
        ],
    )
    def test_primary_tm_accuracy(
        self, reference_results: list[dict[str, Any]], curve_index: int, curve_id: str
    ) -> None:
        """Primary Tm must match DSFworld ground truth within 1.0 C."""
        entry = reference_results[curve_index]
        result = entry["result"]
        expected_tm1 = entry["expected_tm1"]

        assert result.is_valid, f"{curve_id}: fitting failed"
        assert not np.isnan(result.tm), f"{curve_id}: Tm is NaN"

        # For two-transition models, compare sorted Tms
        if entry["expected_tm2"] is not None and result.tm_secondary is not None:
            actual_sorted = sorted([result.tm, result.tm_secondary])
            expected_sorted = sorted([expected_tm1, entry["expected_tm2"]])
            error = abs(actual_sorted[0] - expected_sorted[0])
        else:
            error = abs(result.tm - expected_tm1)

        assert error <= TM_TOLERANCE, (
            f"{curve_id}: Tm1 error = {error:.4f} C "
            f"(expected {expected_tm1:.2f} C, got {result.tm:.2f} C, "
            f"tolerance {TM_TOLERANCE} C)"
        )

    @pytest.mark.parametrize(
        "curve_index, curve_id",
        [
            (3, "DSFbase01_noncanon_ID004714"),
            (4, "DSFbase01_noncanon_ID004711"),
        ],
    )
    def test_secondary_tm_accuracy(
        self, reference_results: list[dict[str, Any]], curve_index: int, curve_id: str
    ) -> None:
        """Secondary Tm for two-transition curves must match within 1.0 C."""
        entry = reference_results[curve_index]
        result = entry["result"]
        expected_tm2 = entry["expected_tm2"]

        assert expected_tm2 is not None, f"{curve_id}: expected_tm2 is None in test data"
        assert result.tm_secondary is not None, (
            f"{curve_id}: BADA did not detect a secondary Tm (expected {expected_tm2:.2f} C)"
        )

        actual_sorted = sorted([result.tm, result.tm_secondary])
        expected_sorted = sorted([entry["expected_tm1"], expected_tm2])
        error = abs(actual_sorted[1] - expected_sorted[1])

        assert error <= TM_TOLERANCE, (
            f"{curve_id}: Tm2 error = {error:.4f} C "
            f"(expected {expected_tm2:.2f} C, got {actual_sorted[1]:.2f} C, "
            f"tolerance {TM_TOLERANCE} C)"
        )


class TestDSFworldReferenceFitQuality:
    """Validate that fits are valid and produce reasonable R-squared values.

    Reference: Wu et al. 2024 [1] — DSFworld achieves 0% failure rate on
    the full 5,747-curve benchmark.
    """

    @pytest.mark.parametrize("curve_index", range(5))
    def test_fit_is_valid(self, reference_results: list[dict[str, Any]], curve_index: int) -> None:
        """Every reference curve must produce a valid fit result."""
        entry = reference_results[curve_index]
        result = entry["result"]
        assert result.is_valid, f"{entry['id']}: fit is not valid"
        assert not np.isnan(result.tm), f"{entry['id']}: Tm is NaN"

    @pytest.mark.parametrize("curve_index", range(5))
    def test_r_squared_above_threshold(
        self, reference_results: list[dict[str, Any]], curve_index: int
    ) -> None:
        """R-squared should be above 0.98 for well-characterized curves."""
        entry = reference_results[curve_index]
        result = entry["result"]
        min_r_squared = 0.98

        assert result.r_squared >= min_r_squared, (
            f"{entry['id']}: R² = {result.r_squared:.6f} (expected >= {min_r_squared})"
        )

    @pytest.mark.parametrize(
        "curve_index, curve_id",
        [
            (3, "DSFbase01_noncanon_ID004714"),
            (4, "DSFbase01_noncanon_ID004711"),
        ],
    )
    def test_two_transition_detected(
        self, reference_results: list[dict[str, Any]], curve_index: int, curve_id: str
    ) -> None:
        """Two-transition curves must produce a secondary Tm."""
        entry = reference_results[curve_index]
        result = entry["result"]
        assert result.tm_secondary is not None, (
            f"{curve_id}: expected two transitions but got only one Tm"
        )
