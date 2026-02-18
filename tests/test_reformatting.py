import numpy as np
import pandas as pd
import pytest

from bada.utils.reformatting import (
    convert_distances_to_plate_format,
    convert_features_to_plate_format,
)


class TestReformatting:
    def test_convert_distances_to_plate_format_96_well(self):
        """Test converting distances to plate format for a 96-well plate."""
        # Create sample distances dictionary
        distances = {
            "A1": (0.0, "A1"),  # (distance, reference_well)
            "A2": (0.1, "A1"),
            "B1": (0.2, "A1"),
            "H12": (1.5, "A1"),
        }

        plate_data, cols, rows = convert_distances_to_plate_format(distances, 96)

        # Check output dimensions
        assert plate_data.shape == (8, 12)  # 8 rows x 12 columns for 96-well plate
        assert len(cols) == 12
        assert len(rows) == 8

        # Check specific values
        assert plate_data[0, 0] == 0.0  # A1
        assert plate_data[0, 1] == 0.1  # A2
        assert plate_data[1, 0] == 0.2  # B1
        assert plate_data[7, 11] == 1.5  # H12

        # Check NaN values for wells not in the dictionary
        assert np.isnan(plate_data[0, 2])  # A3 not in the dictionary

        # Check row and column labels
        assert rows == list("ABCDEFGH")
        assert cols == [str(i) for i in range(1, 13)]

    def test_convert_distances_to_plate_format_384_well(self):
        """Test converting distances to plate format for a 384-well plate."""
        # Create sample distances dictionary
        distances = {
            "A1": (0.0, "A1"),
            "A24": (0.5, "A1"),
            "P1": (0.8, "A1"),
            "P24": (1.2, "A1"),
        }

        plate_data, cols, rows = convert_distances_to_plate_format(distances, 384)

        # Check output dimensions
        assert plate_data.shape == (16, 24)  # 16 rows x 24 columns for 384-well plate
        assert len(cols) == 24
        assert len(rows) == 16

        # Check specific values
        assert plate_data[0, 0] == 0.0  # A1
        assert plate_data[0, 23] == 0.5  # A24
        assert plate_data[15, 0] == 0.8  # P1
        assert plate_data[15, 23] == 1.2  # P24

        # Check row and column labels
        assert rows == list("ABCDEFGHIJKLMNOP")
        assert cols == [str(i) for i in range(1, 25)]

    def test_convert_distances_to_plate_format_invalid_wells(self):
        """Test handling of invalid well positions."""
        # Create sample distances dictionary with a well that doesn't match plate size
        distances = {
            "A1": (0.0, "A1"),
            "I1": (0.2, "A1"),  # Row I doesn't exist in a 96-well plate
        }

        # This should raise a ValueError or IndexError
        with pytest.raises((ValueError, IndexError)):
            convert_distances_to_plate_format(distances, 96)

    def test_convert_features_to_plate_format_96_well(self):
        """Test converting feature data to plate format for a 96-well plate."""
        # Create sample feature data with the correct type annotation
        features: dict[str, dict[str, float | pd.DataFrame | np.ndarray]] = {
            "A1": {"tm": 60.0, "max_derivative_value": 0.5},
            "A2": {"tm": 65.0, "max_derivative_value": 0.6},
            "B1": {"tm": 70.0, "max_derivative_value": 0.7},
            "H12": {"tm": 72.0, "max_derivative_value": 0.8},
        }

        # Test with tm feature
        plate_data, cols, rows = convert_features_to_plate_format(features, 96, "tm")

        # Check output dimensions
        assert plate_data.shape == (8, 12)
        assert len(cols) == 12
        assert len(rows) == 8

        # Check specific values
        assert plate_data[0, 0] == 60.0  # A1 tm
        assert plate_data[0, 1] == 65.0  # A2 tm
        assert plate_data[1, 0] == 70.0  # B1 tm
        assert plate_data[7, 11] == 72.0  # H12 tm

        # Test with max_derivative_value feature
        plate_data, _, _ = convert_features_to_plate_format(features, 96, "max_derivative_value")

        # Check specific values
        assert plate_data[0, 0] == 0.5  # A1 max_derivative_value
        assert plate_data[0, 1] == 0.6  # A2 max_derivative_value
        assert plate_data[1, 0] == 0.7  # B1 max_derivative_value
        assert plate_data[7, 11] == 0.8  # H12 max_derivative_value

    def test_convert_features_to_plate_format_raises_keyerror_for_missing_feature(self):
        """Test that convert_features_to_plate_format raises KeyError when a requested feature is
        missing."""
        # Create sample feature data with missing feature for one well
        features: dict[str, dict[str, float | pd.DataFrame | np.ndarray]] = {
            "A1": {"tm": 60.0, "max_derivative_value": 0.5},
            "A2": {"tm": 65.0},  # missing max_derivative_value
        }

        # Function should raise KeyError since the feature is missing for one well
        with pytest.raises(KeyError):
            convert_features_to_plate_format(features, 96, "max_derivative_value")

    def test_convert_features_to_plate_format_raises_keyerror_for_nonexistent_feature(self):
        """Test that convert_features_to_plate_format raises KeyError when a nonexistent feature is
        requested."""
        # Create sample feature data
        features: dict[str, dict[str, float | pd.DataFrame | np.ndarray]] = {
            "A1": {"tm": 60.0, "max_derivative_value": 0.5},
            "A2": {"tm": 65.0, "max_derivative_value": 0.6},
        }

        # Function should raise KeyError for nonexistent feature
        with pytest.raises(KeyError):
            convert_features_to_plate_format(features, 96, "nonexistent_feature")
