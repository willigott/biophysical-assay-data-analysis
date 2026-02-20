from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import BSpline, make_splrep

from bada.models.dsf_input_model import DSFInput


@pytest.fixture
def sample_temperatures() -> np.ndarray:
    """Sample temperature data for testing."""
    return np.linspace(25, 95, 100)


@pytest.fixture
def sample_fluorescence() -> np.ndarray:
    """Sample fluorescence data using a Boltzmann sigmoid (two-state unfolding model).

    Generates a realistic DSF melting curve with known Tm=55.0°C.
    The Boltzmann model f(T) = f_n + (f_d - f_n) / (1 + exp((Tm - T) / slope))
    has its first derivative maximum at exactly T=Tm, providing a ground truth
    for Tm detection accuracy.

    Reference: Niesen et al. 2007, Nature Protocols 2(9), 2212-2221. [10]
    """
    rng = np.random.default_rng(42)
    x = np.linspace(25, 95, 100)
    f_native = 500.0
    f_denatured = 5000.0
    tm = 55.0
    slope = 2.0
    y = f_native + (f_denatured - f_native) / (1.0 + np.exp((tm - x) / slope))
    y = y + rng.normal(0, 5.0, len(x))
    return y


@pytest.fixture
def sample_dsf_data(
    sample_temperatures: np.ndarray, sample_fluorescence: np.ndarray
) -> pd.DataFrame:
    """Sample DSF data in DataFrame format for testing."""
    rng = np.random.default_rng(43)
    data = []

    # First well
    for temp, fluor in zip(sample_temperatures, sample_fluorescence):
        data.append({"well_position": "A1", "temperature": temp, "fluorescence": fluor})

    fluor2 = sample_fluorescence + rng.normal(0, 0.5, len(sample_fluorescence))
    for temp, fluor in zip(sample_temperatures, fluor2):
        data.append({"well_position": "A2", "temperature": temp, "fluorescence": fluor})

    return pd.DataFrame(data)


@pytest.fixture
def sample_validated_dsf_data(sample_dsf_data: pd.DataFrame) -> pd.DataFrame:
    """Sample DSF data that passes validation."""
    return DSFInput.validate(sample_dsf_data)


@pytest.fixture
def sample_spline(sample_temperatures: np.ndarray, sample_fluorescence: np.ndarray) -> BSpline:
    """Sample spline for testing."""
    return make_splrep(sample_temperatures, sample_fluorescence, s=0.01)  # type: ignore


@pytest.fixture
def sample_spline_x(sample_temperatures: np.ndarray) -> np.ndarray:
    """Sample x values for spline."""
    return np.linspace(min(sample_temperatures), max(sample_temperatures), 1000)


@pytest.fixture
def mock_file_path(mocker) -> Any:
    """Mock file path for testing parsers."""
    return mocker.Mock(spec=Path)


@pytest.fixture
def sample_plate_data() -> np.ndarray:
    """Sample plate data for heatmap visualization."""
    rng = np.random.default_rng(44)
    # 8x12 plate (rows A-H, columns 1-12)
    plate = rng.normal(50, 10, size=(8, 12))
    return plate


@pytest.fixture
def plate_rows() -> list[str]:
    """Row labels for a standard 96-well plate."""
    return list("ABCDEFGH")


@pytest.fixture
def plate_cols() -> list[str]:
    """Column labels for a standard 96-well plate."""
    return [str(i) for i in range(1, 13)]


def _boltzmann_sigmoid(
    x: np.ndarray,
    f_native: float,
    f_denatured: float,
    tm: float,
    slope: float,
) -> np.ndarray:
    """Boltzmann sigmoid: f(T) = f_n + (f_d - f_n) / (1 + exp((Tm - T) / slope))."""
    return f_native + (f_denatured - f_native) / (1.0 + np.exp((tm - x) / slope))


@pytest.fixture
def lysozyme_like_fluorescence() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic DSF curve mimicking hen egg white lysozyme (HEWL).

    Tm=72.0°C with slight linear baseline drift (0.5 RFU/°C instrument thermal drift).
    Lysozyme is a small, highly cooperative folder (slope=1.5).

    Published values: Tm=72.4°C at pH 6.0 (Schönfelder et al. 2025 [12]);
    Tm≈69°C at pH 7.2 (Wu et al. 2023 [9]).

    Returns:
        Tuple of (temperature, fluorescence) arrays.
    """
    rng = np.random.default_rng(45)
    x = np.linspace(25, 95, 141)
    y = _boltzmann_sigmoid(x, f_native=200.0, f_denatured=8000.0, tm=72.0, slope=1.5)
    # Linear baseline drift
    y = y + 0.5 * (x - 25.0)
    y = y + rng.normal(0, 10.0, len(x))
    return x, y


@pytest.fixture
def multi_domain_fluorescence() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic DSF curve for a two-domain protein with distinct transitions.

    Transition 1: Tm1=50.0°C (amplitude 2000, slope 2.5)
    Transition 2: Tm2=65.0°C (amplitude 3000, slope 2.0)

    Multi-domain proteins are common in drug target families. BSA has 3 domains
    with Tm1≈56°C, Tm2≈62°C, Tm3≈72°C at pH 7 (Gao et al. 2020 [8]).

    Returns:
        Tuple of (temperature, fluorescence) arrays.
    """
    rng = np.random.default_rng(46)
    x = np.linspace(25, 95, 141)
    y = 400.0
    y = y + _boltzmann_sigmoid(x, f_native=0.0, f_denatured=2000.0, tm=50.0, slope=2.5)
    y = y + _boltzmann_sigmoid(x, f_native=0.0, f_denatured=3000.0, tm=65.0, slope=2.0)
    y = y + rng.normal(0, 10.0, len(x))
    return x, y


@pytest.fixture
def high_initial_fluorescence() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic DSF curve with high initial fluorescence and exponential decay.

    This is the most common non-canonical DSF curve type, accounting for 41.2%
    of curves in the DSFworld benchmark (Model 2) (Wu et al. 2024 [1]).

    Sigmoid (Tm=60°C) + initial decay: 3000 * exp(-0.05 * (T-25)).

    Returns:
        Tuple of (temperature, fluorescence) arrays.
    """
    rng = np.random.default_rng(47)
    x = np.linspace(25, 95, 141)
    y = _boltzmann_sigmoid(x, f_native=500.0, f_denatured=4000.0, tm=60.0, slope=2.0)
    y = y + 3000.0 * np.exp(-0.05 * (x - 25.0))
    y = y + rng.normal(0, 10.0, len(x))
    return x, y


@pytest.fixture
def aggregation_artifact_fluorescence() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic DSF curve with post-Tm aggregation artifact.

    After unfolding (Tm=58°C), the fluorescence drops due to protein aggregation
    (centered at 75°C), creating a spurious peak in the derivative.

    Reference: Gao et al. 2020 [8] — aggregation is a well-documented complication.

    Returns:
        Tuple of (temperature, fluorescence) arrays.
    """
    rng = np.random.default_rng(48)
    x = np.linspace(25, 95, 141)
    # Unfolding transition
    y = _boltzmann_sigmoid(x, f_native=500.0, f_denatured=6000.0, tm=58.0, slope=2.0)
    # Aggregation: fluorescence drops after unfolding
    y = y - _boltzmann_sigmoid(x, f_native=0.0, f_denatured=2000.0, tm=75.0, slope=3.0)
    y = y + rng.normal(0, 10.0, len(x))
    return x, y


@pytest.fixture
def flat_fluorescence() -> tuple[np.ndarray, np.ndarray]:
    """Flat fluorescence trace — no unfolding transition.

    Represents an empty well, pre-denatured protein, or instrument failure.
    Uses a constant signal so that the spline derivative is effectively zero,
    triggering flat derivative detection.

    Reference: Sun et al. 2020 [4] classifies flat traces as "no transition";
    Niesen et al. 2007 [10] describes flat traces as a common DSF failure mode.

    Returns:
        Tuple of (temperature, fluorescence) arrays.
    """
    x = np.linspace(25, 95, 141)
    y = np.full_like(x, 1000.0)
    return x, y


@pytest.fixture
def edge_artifact_fluorescence() -> tuple[np.ndarray, np.ndarray]:
    """DSF curve with a strong artifact at the high-temperature boundary.

    Real transition at 60°C; spurious spike near 93°C from dye degradation.

    Reference: Wu et al. 2023 [9] — recommends excluding scan endpoints.

    Returns:
        Tuple of (temperature, fluorescence) arrays.
    """
    rng = np.random.default_rng(50)
    x = np.linspace(25, 95, 141)
    y = _boltzmann_sigmoid(x, f_native=500.0, f_denatured=5000.0, tm=60.0, slope=2.0)
    # Strong artifact at high-temperature boundary
    artifact = 4000.0 * np.exp(-((x - 93.0) ** 2) / 2.0)
    y = y + artifact
    y = y + rng.normal(0, 5.0, len(x))
    return x, y
