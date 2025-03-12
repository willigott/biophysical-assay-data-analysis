from pathlib import Path
from typing import Any, List

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
    """Sample fluorescence data for testing."""
    # Create a smooth peak-like signal with minimal noise
    x = np.linspace(25, 95, 100)
    # Reduced noise level from 10 to 2
    y = 1000 + 500 * np.exp(-((x - 65) ** 2) / 50) + np.random.normal(0, 0.2, 100)
    return y


@pytest.fixture
def sample_dsf_data(
    sample_temperatures: np.ndarray, sample_fluorescence: np.ndarray
) -> pd.DataFrame:
    """Sample DSF data in DataFrame format for testing."""
    # Create a simple dataframe with data for two wells
    data = []

    # First well
    for i, (temp, fluor) in enumerate(zip(sample_temperatures, sample_fluorescence)):
        data.append({"well_position": "A1", "temperature": temp, "fluorescence": fluor})

    fluor2 = sample_fluorescence + np.random.normal(0, 0.5, 100)
    for i, (temp, fluor) in enumerate(zip(sample_temperatures, fluor2)):
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
    # 8x12 plate (rows A-H, columns 1-12)
    plate = np.random.normal(50, 10, size=(8, 12))
    return plate


@pytest.fixture
def plate_rows() -> List[str]:
    """Row labels for a standard 96-well plate."""
    return list("ABCDEFGH")


@pytest.fixture
def plate_cols() -> List[str]:
    """Column labels for a standard 96-well plate."""
    return [str(i) for i in range(1, 13)]
