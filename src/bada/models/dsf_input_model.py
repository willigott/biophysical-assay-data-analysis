import pandera as pa
from pandera.typing import Series


class DSFInput(pa.DataFrameModel):
    well_position: Series[str] = pa.Field(str_matches=r"^[A-P](?:[1-9]|1[0-9]|2[0-4])$")
    temperature: Series[float] = pa.Field(ge=0, le=100)
    fluorescence: Series[float] = pa.Field()
