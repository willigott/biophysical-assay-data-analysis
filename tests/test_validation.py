import pytest

from bada.utils.validation import validate_temperature_range


class TestValidation:
    def test_validate_temperature_range_valid(self):
        """Test that validate_temperature_range returns True for valid temperature ranges."""
        # Valid temperature range
        assert validate_temperature_range(25.0, 95.0) is True

        # Edge case: very close temperatures
        assert validate_temperature_range(25.0, 25.1) is True

        # Edge case: negative temperatures (valid as long as min < max)
        assert validate_temperature_range(-10.0, 10.0) is True

    def test_validate_temperature_range_invalid(self):
        """Test that validate_temperature_range raises ValueError for invalid temperature ranges."""
        # Min temperature equals max temperature
        with pytest.raises(ValueError):
            validate_temperature_range(25.0, 25.0)

        # Min temperature greater than max temperature
        with pytest.raises(ValueError):
            validate_temperature_range(50.0, 25.0)

        # Edge case: slightly greater
        with pytest.raises(ValueError):
            validate_temperature_range(25.01, 25.0)

    def test_validate_temperature_range_none_values(self):
        """Test that validate_temperature_range handles None values correctly."""
        # Both None
        assert validate_temperature_range(None, None) is False

        # One None
        assert validate_temperature_range(25.0, None) is False
        assert validate_temperature_range(None, 95.0) is False
