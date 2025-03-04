from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseParser(ABC):
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def _validate_path(self) -> None:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")

    @abstractmethod
    def parse(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _read_file(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _validate_raw_data(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def _process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def _validate_processed_data(self, df: pd.DataFrame) -> None:
        pass
