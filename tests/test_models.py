import pandas as pd
import pytest

from bada.models.dsf_input_model import DSFInput
from bada.models.lightcycler480 import LightCycler480Raw
from bada.models.quantstudio7 import QuantStudio7Raw


class TestDSFInput:
    def test_valid_data_passes_validation(self, sample_dsf_data: pd.DataFrame) -> None:
        """Test that valid data passes DSFInput validation."""
        # This should not raise an exception
        validated_data = DSFInput.validate(sample_dsf_data)
        assert isinstance(validated_data, pd.DataFrame)
        assert "well_position" in validated_data.columns
        assert "temperature" in validated_data.columns
        assert "fluorescence" in validated_data.columns

    def test_invalid_well_position_fails_validation(self, sample_dsf_data: pd.DataFrame) -> None:
        """Test that invalid well positions fail validation."""
        # Create data with invalid well position
        invalid_data = sample_dsf_data.copy()
        invalid_data.loc[0, "well_position"] = "Z1"  # Not a valid well

        with pytest.raises(Exception):
            DSFInput.validate(invalid_data)

    def test_invalid_temperature_fails_validation(self, sample_dsf_data: pd.DataFrame) -> None:
        """Test that invalid temperatures fail validation."""
        # Create data with invalid temperature (negative value)
        invalid_data = sample_dsf_data.copy()
        invalid_data.loc[0, "temperature"] = -10

        with pytest.raises(Exception):
            DSFInput.validate(invalid_data)

        # Create data with invalid temperature (too high)
        invalid_data = sample_dsf_data.copy()
        invalid_data.loc[0, "temperature"] = 101

        with pytest.raises(Exception):
            DSFInput.validate(invalid_data)


class TestLightCycler480Raw:
    def test_attributes(self) -> None:
        """Test that LightCycler480Raw has the expected attributes."""
        assert hasattr(LightCycler480Raw, "temperature")
        assert hasattr(LightCycler480Raw, "_well")
        assert hasattr(LightCycler480Raw, "_x_n")


class TestQuantStudio7Raw:
    def test_attributes(self) -> None:
        """Test that QuantStudio7Raw has the expected attributes."""
        assert hasattr(QuantStudio7Raw, "Well")
        assert hasattr(QuantStudio7Raw, "well_position")
        assert hasattr(QuantStudio7Raw, "Temperature")
        assert hasattr(QuantStudio7Raw, "Fluorescence")
