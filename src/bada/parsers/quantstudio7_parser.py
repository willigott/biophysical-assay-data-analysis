from pathlib import Path

import pandas as pd

from bada.models import DSFInput, QuantStudio7Raw
from bada.parsers.base_parser import BaseParser

DEFAULT_SKIPROWS = 21


class QuantStudio7Parser(BaseParser):
    def __init__(self, file_path: Path, skiprows: int = DEFAULT_SKIPROWS):
        super().__init__(file_path)
        self.skiprows = skiprows

    def _read_file(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path, skiprows=self.skiprows)

    def _validate_raw_data(self, df: pd.DataFrame) -> None:
        QuantStudio7Raw.validate(df)

    def _process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(
            columns={
                "Well Position": "well_position",
                "Temperature": "temperature",
                "Fluorescence": "fluorescence",
            }
        )

        df = df.loc[:, ["well_position", "temperature", "fluorescence"]]

        return df

    def _validate_processed_data(self, df: pd.DataFrame) -> None:
        DSFInput.validate(df)
