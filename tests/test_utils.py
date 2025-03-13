import pytest

from bada.utils.utils import get_column_labels, get_row_labels, get_well_ids


class TestUtils:
    def test_get_row_labels_96_well(self):
        """Test that get_row_labels returns correct rows for 96-well plate."""
        rows = get_row_labels(96)
        assert rows == list("ABCDEFGH")
        assert len(rows) == 8

    def test_get_row_labels_384_well(self):
        """Test that get_row_labels returns correct rows for 384-well plate."""
        rows = get_row_labels(384)
        assert rows == list("ABCDEFGHIJKLMNOP")
        assert len(rows) == 16

    def test_get_row_labels_invalid_plate_size(self):
        """Test that get_row_labels raises error for invalid plate size."""
        with pytest.raises(ValueError):
            get_row_labels(12)

        with pytest.raises(ValueError):
            get_row_labels(192)

    def test_get_column_labels_96_well_as_str(self):
        """Test that get_column_labels returns correct columns for 96-well plate as strings."""
        cols = get_column_labels(96, as_str=True)
        assert cols == [str(i) for i in range(1, 13)]
        assert len(cols) == 12
        assert all(isinstance(c, str) for c in cols)

    def test_get_column_labels_96_well_as_int(self):
        """Test that get_column_labels returns correct columns for 96-well plate as integers."""
        cols = get_column_labels(96, as_str=False)
        assert cols == list(range(1, 13))
        assert len(cols) == 12
        assert all(isinstance(c, int) for c in cols)

    def test_get_column_labels_384_well_as_str(self):
        """Test that get_column_labels returns correct columns for 384-well plate as strings."""
        cols = get_column_labels(384, as_str=True)
        assert cols == [str(i) for i in range(1, 25)]
        assert len(cols) == 24
        assert all(isinstance(c, str) for c in cols)

    def test_get_column_labels_384_well_as_int(self):
        """Test that get_column_labels returns correct columns for 384-well plate as integers."""
        cols = get_column_labels(384, as_str=False)
        assert cols == list(range(1, 25))
        assert len(cols) == 24
        assert all(isinstance(c, int) for c in cols)

    def test_get_column_labels_invalid_plate_size(self):
        """Test that get_column_labels raises error for invalid plate size."""
        with pytest.raises(ValueError):
            get_column_labels(12)

        with pytest.raises(ValueError):
            get_column_labels(192)

    def test_get_well_ids_96_well(self):
        """Test that get_well_ids returns correct well IDs for 96-well plate."""
        wells = get_well_ids(96)
        assert len(wells) == 96  # 8 rows x 12 columns

        # Check first few wells
        assert wells[0] == "A1"
        assert wells[1] == "A2"
        assert wells[11] == "A12"
        assert wells[12] == "B1"

        # Verify expected patterns
        for i, well in enumerate(wells):
            row_idx = i // 12
            col_idx = i % 12 + 1
            assert well == f"{chr(65 + row_idx)}{col_idx}"

    def test_get_well_ids_384_well(self):
        """Test that get_well_ids returns correct well IDs for 384-well plate."""
        wells = get_well_ids(384)
        assert len(wells) == 384  # 16 rows x 24 columns

        # Check first few wells
        assert wells[0] == "A1"
        assert wells[1] == "A2"
        assert wells[23] == "A24"
        assert wells[24] == "B1"

        # Verify expected patterns
        for i, well in enumerate(wells):
            row_idx = i // 24
            col_idx = i % 24 + 1
            assert well == f"{chr(65 + row_idx)}{col_idx}"

    def test_get_well_ids_invalid_plate_size(self):
        """Test that get_well_ids raises error for invalid plate size."""
        with pytest.raises(ValueError):
            get_well_ids(12)

        with pytest.raises(ValueError):
            get_well_ids(192)
