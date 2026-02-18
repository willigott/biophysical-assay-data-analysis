from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bada.parsers.base_parser import BaseParser
from bada.parsers.lightcycler480_parser import LightCycler480Parser
from bada.parsers.quantstudio7_parser import QuantStudio7Parser


class TestBaseParser:
    class ConcreteParser(BaseParser):
        """Concrete implementation of the abstract BaseParser for testing."""

        def _read_file(self) -> pd.DataFrame:
            return pd.DataFrame()

        def _validate_raw_data(self, df: pd.DataFrame) -> None:
            pass

        def _process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

        def _validate_processed_data(self, df: pd.DataFrame) -> None:
            pass

    def test_init(self, mock_file_path: Any) -> None:
        """Test BaseParser initialization."""
        parser = self.ConcreteParser(mock_file_path)
        assert parser.file_path == mock_file_path

    def test_validate_path_file_not_found(self, mock_file_path: Any) -> None:
        """Test _validate_path when file doesn't exist."""
        mock_file_path.exists.return_value = False
        parser = self.ConcreteParser(mock_file_path)

        with pytest.raises(FileNotFoundError):
            parser._validate_path()

    def test_validate_path_not_a_file(self, mock_file_path: Any) -> None:
        """Test _validate_path when path is not a file."""
        mock_file_path.exists.return_value = True
        mock_file_path.is_file.return_value = False
        parser = self.ConcreteParser(mock_file_path)

        with pytest.raises(ValueError):
            parser._validate_path()

    def test_validate_path_success(self, mock_file_path: Any) -> None:
        """Test _validate_path when path is valid."""
        mock_file_path.exists.return_value = True
        mock_file_path.is_file.return_value = True
        parser = self.ConcreteParser(mock_file_path)

        # This should not raise an exception
        parser._validate_path()

    def test_abstract_methods(self, mock_file_path):
        """Test that abstract hook methods need to be implemented.

        BaseParser.parse() is a concrete template method. Only the four hook
        methods (_read_file, _validate_raw_data, _process_raw_data,
        _validate_processed_data) are abstract and must be provided by
        subclasses.

        Reference: Template Method pattern — GoF Design Patterns, Chapter 5.
        """
        from bada.parsers.base_parser import BaseParser

        # BaseParser itself cannot be instantiated (has abstract hooks)
        with pytest.raises(TypeError):
            BaseParser(mock_file_path)  # type: ignore

        # Missing _read_file
        class MissingReadFile(BaseParser):
            def _validate_raw_data(self, df):
                pass

            def _process_raw_data(self, df):
                pass

            def _validate_processed_data(self, df):
                pass

        with pytest.raises(TypeError):
            MissingReadFile(mock_file_path)  # type: ignore

        # Missing _validate_raw_data
        class MissingValidateRaw(BaseParser):
            def _read_file(self):
                pass

            def _process_raw_data(self, df):
                pass

            def _validate_processed_data(self, df):
                pass

        with pytest.raises(TypeError):
            MissingValidateRaw(mock_file_path)  # type: ignore

        # Missing _process_raw_data
        class MissingProcessRaw(BaseParser):
            def _read_file(self):
                pass

            def _validate_raw_data(self, df):
                pass

            def _validate_processed_data(self, df):
                pass

        with pytest.raises(TypeError):
            MissingProcessRaw(mock_file_path)  # type: ignore

        # Missing _validate_processed_data
        class MissingValidateProcessed(BaseParser):
            def _read_file(self):
                pass

            def _validate_raw_data(self, df):
                pass

            def _process_raw_data(self, df):
                pass

        with pytest.raises(TypeError):
            MissingValidateProcessed(mock_file_path)  # type: ignore

    def test_parse_calls_hooks_in_order(self, mocker, mock_file_path: Any) -> None:
        """Test that the template method calls hooks in the correct sequence.

        The Template Method pattern defines an algorithm skeleton in the base
        class, deferring steps to subclasses. This test verifies parse()
        orchestrates the four hooks in order: _read_file -> _validate_raw_data
        -> _process_raw_data -> _validate_processed_data.

        Reference: Template Method pattern — GoF Design Patterns, Chapter 5.
        """
        parser = self.ConcreteParser(mock_file_path)

        call_order: list[str] = []

        mock_df = pd.DataFrame({"col": [1]})

        mocker.patch.object(
            parser, "_read_file", side_effect=lambda: (call_order.append("_read_file"), mock_df)[1]
        )
        mocker.patch.object(
            parser,
            "_validate_raw_data",
            side_effect=lambda df: call_order.append("_validate_raw_data"),
        )
        mocker.patch.object(
            parser,
            "_process_raw_data",
            side_effect=lambda df: (call_order.append("_process_raw_data"), mock_df)[1],
        )
        mocker.patch.object(
            parser,
            "_validate_processed_data",
            side_effect=lambda df: call_order.append("_validate_processed_data"),
        )

        result = parser.parse()

        assert call_order == [
            "_read_file",
            "_validate_raw_data",
            "_process_raw_data",
            "_validate_processed_data",
        ]
        assert isinstance(result, pd.DataFrame)


class TestLightCycler480Parser:
    def test_read_file(self, mocker, mock_file_path: Any) -> None:
        """Test that _read_file calls pd.read_csv with the right parameters."""
        mock_read_csv = mocker.patch("bada.parsers.lightcycler480_parser.pd.read_csv")
        parser = LightCycler480Parser(mock_file_path)
        parser._read_file()
        mock_read_csv.assert_called_once_with(mock_file_path)

    def test_parse(self, mocker, mock_file_path: Any) -> None:
        """Test the parse method calls the expected methods in sequence."""
        # Setup mocks
        mock_read = mocker.patch.object(LightCycler480Parser, "_read_file")
        mock_validate_raw = mocker.patch.object(LightCycler480Parser, "_validate_raw_data")
        mock_process = mocker.patch.object(LightCycler480Parser, "_process_raw_data")
        mock_validate_processed = mocker.patch.object(
            LightCycler480Parser, "_validate_processed_data"
        )

        # Return values
        mock_read.return_value = pd.DataFrame()
        mock_process.return_value = pd.DataFrame()

        # Execute
        parser = LightCycler480Parser(mock_file_path)
        result = parser.parse()

        # Verify
        mock_read.assert_called_once()
        mock_validate_raw.assert_called_once()
        mock_process.assert_called_once()
        mock_validate_processed.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    def test_process_raw_data(self):
        """Test that _process_raw_data transforms data correctly."""
        from bada.parsers.lightcycler480_parser import LightCycler480Parser

        # Create a mock raw dataframe in the LightCycler480 format
        raw_data = pd.DataFrame(
            {
                "A1: Temperature": [25.0, 30.0, 35.0],
                "A1: Fluorescence": [100.0, 110.0, 120.0],
                "A2: Temperature": [25.0, 30.0, 35.0],
                "A2: Fluorescence": [105.0, 115.0, 125.0],
            }
        )

        parser = LightCycler480Parser(Path("dummy_path"))
        processed_data = parser._process_raw_data(raw_data)

        # Check resulting dataframe has the correct format
        assert list(processed_data.columns) == ["temperature", "well_position", "fluorescence"]
        assert len(processed_data) == 6  # 3 temperature points x 2 wells
        assert set(processed_data["well_position"].unique()) == {"A1", "A2"}

        # Check data values are preserved
        a1_data = processed_data[processed_data["well_position"] == "A1"]
        assert list(a1_data["temperature"]) == [25.0, 30.0, 35.0]
        assert list(a1_data["fluorescence"]) == [100.0, 110.0, 120.0]


class TestQuantStudio7Parser:
    def test_read_file_default_skiprows(self, mocker, mock_file_path: Any) -> None:
        """Test that _read_file uses default skiprows=21."""
        mock_read_csv = mocker.patch("bada.parsers.quantstudio7_parser.pd.read_csv")
        parser = QuantStudio7Parser(mock_file_path)
        parser._read_file()
        mock_read_csv.assert_called_once_with(mock_file_path, skiprows=21)

    def test_read_file_custom_skiprows(self, mocker, mock_file_path: Any) -> None:
        """Test that _read_file respects a custom skiprows value."""
        mock_read_csv = mocker.patch("bada.parsers.quantstudio7_parser.pd.read_csv")
        parser = QuantStudio7Parser(mock_file_path, skiprows=10)
        parser._read_file()
        mock_read_csv.assert_called_once_with(mock_file_path, skiprows=10)

    def test_parse(self, mocker, mock_file_path: Any) -> None:
        """Test the parse method calls the expected methods in sequence."""
        # Setup mocks
        mock_read = mocker.patch.object(QuantStudio7Parser, "_read_file")
        mock_validate_raw = mocker.patch.object(QuantStudio7Parser, "_validate_raw_data")
        mock_process = mocker.patch.object(QuantStudio7Parser, "_process_raw_data")
        mock_validate_processed = mocker.patch.object(
            QuantStudio7Parser, "_validate_processed_data"
        )

        # Return values
        mock_read.return_value = pd.DataFrame()
        mock_process.return_value = pd.DataFrame()

        # Execute
        parser = QuantStudio7Parser(mock_file_path)
        result = parser.parse()

        # Verify
        mock_read.assert_called_once()
        mock_validate_raw.assert_called_once()
        mock_process.assert_called_once()
        mock_validate_processed.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    def test_process_raw_data(self):
        """Test that _process_raw_data transforms data correctly."""
        from bada.parsers.quantstudio7_parser import QuantStudio7Parser

        # Create a mock raw dataframe in the QuantStudio7 format
        raw_data = pd.DataFrame(
            {
                "Well Position": ["A1", "A1", "A2", "A2"],
                "Temperature": [25.0, 30.0, 25.0, 30.0],
                "Fluorescence": [100.0, 110.0, 105.0, 115.0],
                "Extra Column": ["x", "y", "z", "w"],  # This should be removed
            }
        )

        parser = QuantStudio7Parser(Path("dummy_path"))
        processed_data = parser._process_raw_data(raw_data)

        # Check resulting dataframe has the correct format
        assert list(processed_data.columns) == ["well_position", "temperature", "fluorescence"]
        assert len(processed_data) == 4
        assert set(processed_data["well_position"].unique()) == {"A1", "A2"}

        # Check data values are preserved and extra column removed
        assert "Extra Column" not in processed_data.columns
        a1_data = processed_data[processed_data["well_position"] == "A1"]
        assert list(a1_data["temperature"]) == [25.0, 30.0]
        assert list(a1_data["fluorescence"]) == [100.0, 110.0]
