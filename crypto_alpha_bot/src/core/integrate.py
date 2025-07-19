import pandas as pd
from typing import Dict
import logging

LOG = logging.getLogger("integrate")

def merge_derivatives_into_price(df: pd.DataFrame,
                                 funding_df: pd.DataFrame | None,
                                 oi_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Usa merge_asof para alinhar funding (cada 8h) e OI (1h) aos candles por open_time.
    Pressupõe df com coluna open_time (ms).
    """
    if df.empty:
        return df

    out = df.copy()

    if funding_df is not None and not funding_df.empty:
        f = funding_df.copy().sort_values("funding_time")
        out = pd.merge_asof(
            out.sort_values("open_time"),
            f.rename(columns={"funding_rate": "funding"}),
            left_on="open_time",
            right_on="funding_time",
            direction="backward"
        )
    if oi_df is not None and not oi_df.empty:
        o = oi_df.copy().sort_values("timestamp")
        out = pd.merge_asof(
            out.sort_values("open_time"),
            o,
            left_on="open_time",
            right_on="timestamp",
            direction="backward"
        )
    return out
