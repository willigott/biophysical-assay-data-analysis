import pandas as pd

from bada.models import DSFInput, LightCycler480Raw
from bada.parsers.base_parser import BaseParser


class LightCycler480Parser(BaseParser):
    def _read_file(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path)

    def _validate_raw_data(self, df: pd.DataFrame) -> None:
        LightCycler480Raw.validate(df)

    def _process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data into a standardized format"""

        temp_cols = df.columns[::2]
        fluor_cols = df.columns[1::2]

        temp_df = df[temp_cols].melt(var_name="well_position", value_name="temperature")
        fluor_df = df[fluor_cols].melt(var_name="well_position", value_name="fluorescence")

        df_stacked = pd.concat(
            [temp_df["temperature"], fluor_df[["well_position", "fluorescence"]]], axis=1
        )

        df_stacked["well_position"] = df_stacked["well_position"].str.strip().str.split(":").str[0]

        return df_stacked

    def _validate_processed_data(self, df: pd.DataFrame) -> None:
        DSFInput.validate(df)
