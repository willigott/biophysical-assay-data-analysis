import pandera as pa
from pandera.typing import Series


class LightCycler480Raw(pa.DataFrameModel):
    temperature: Series[float] = pa.Field(alias="1", ge=0, le=100)

    # allow any additional well columns that match the pattern
    _well: Series[float] = pa.Field(
        regex=r"^[A-P](?:[1-9]|1[0-9]|2[0-4]): Sample \d+$",  # type: ignore
        alias=True,
    )

    # allow X.n columns where n is 1-1536
    _x_n: Series[float] = pa.Field(
        regex=r"^X\.(?:[1-9]|[1-9][0-9]|[1-9][0-9][0-9]|1[0-4][0-9][0-9]|15[0-2][0-9]|153[0-6])$",  # type: ignore
        alias=True,
    )
