import pandera as pa
from pandera.typing import Series


class QuantStudio7Raw(pa.DataFrameModel):
    Well: Series[int] = pa.Field(ge=1, le=384)
    well_position: Series[str] = pa.Field(
        alias="Well Position", str_matches=r"^[A-P](?:[1-9]|1[0-9]|2[0-4])$"
    )
    reading_number: Series[int] = pa.Field(alias="Reading Number")
    Target: Series[str] = pa.Field()
    Temperature: Series[float] = pa.Field(ge=0, le=100)
    # ge=0, le=1_000_000, seems we sometimes have negative values...
    Fluorescence: Series[float] = pa.Field()
    Derivative: Series[float] = pa.Field()
