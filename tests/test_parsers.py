from typing import Any

import pandas as pd
import pytest

from bada.parsers.base_parser import BaseParser
from bada.parsers.lightcycler480_parser import LightCycler480Parser
from bada.parsers.quantstudio7_parser import QuantStudio7Parser


class TestBaseParser:
    class ConcreteParser(BaseParser):
        """Concrete implementation of the abstract BaseParser for testing."""

        def parse(self) -> pd.DataFrame:
            return pd.DataFrame()

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


class TestQuantStudio7Parser:
    def test_read_file(self, mocker, mock_file_path: Any) -> None:
        """Test that _read_file calls pd.read_csv with the right parameters."""
        mock_read_csv = mocker.patch("bada.parsers.quantstudio7_parser.pd.read_csv")
        parser = QuantStudio7Parser(mock_file_path)
        parser._read_file()
        mock_read_csv.assert_called_once_with(mock_file_path, skiprows=21)

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
