import pandas as pd
import pathlib

class ParquetStore:
    """
    Armazena incrementalmente OHLCV.
    Fallback para CSV se não houver engine Parquet.
    """

    def __init__(self, base_path: str, symbol: str, timeframe: str):
        self.base = pathlib.Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.timeframe = timeframe
        self.parquet_path = self.base / f"{symbol}_{timeframe}.parquet"
        self.csv_path = self.base / f"{symbol}_{timeframe}.csv"

    def _has_parquet(self) -> bool:
        try:
            import pyarrow  # noqa
            return True
        except Exception:
            try:
                import fastparquet  # noqa
                return True
            except Exception:
                return False

    def load(self) -> pd.DataFrame:
        if self.parquet_path.exists():
            try:
                return pd.read_parquet(self.parquet_path)
            except Exception:
                pass
        if self.csv_path.exists():
            try:
                return pd.read_csv(self.csv_path)
            except Exception:
                pass
        return pd.DataFrame()

    def append_and_dedupe(self, df: pd.DataFrame, key: str = "open_time") -> pd.DataFrame:
        existing = self.load()
        if not existing.empty:
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=[key]).sort_values(key)
        else:
            combined = df.sort_values(key)

        if self._has_parquet():
            combined.to_parquet(self.parquet_path, index=False)
        else:
            combined.to_csv(self.csv_path, index=False)

        return combined
