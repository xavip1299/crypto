import pandas as pd, pathlib
from typing import Tuple

class ParquetStore:
    def __init__(self, base_path: str, symbol: str, timeframe: str):
        self.base = pathlib.Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)
        self.path = self.base / f"{symbol}_{timeframe}.parquet"

    def load(self) -> pd.DataFrame:
        if self.path.exists():
            return pd.read_parquet(self.path)
        return pd.DataFrame()

    def append_and_dedupe(self, df: pd.DataFrame, key="open_time") -> pd.DataFrame:
        existing = self.load()
        if not existing.empty:
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=[key]).sort_values(key)
        else:
            combined = df.sort_values(key)
        combined.to_parquet(self.path, index=False)
        return combined
