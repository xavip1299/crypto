import pandas as pd
import numpy as np

def add_cross_sectional_stats(df: pd.DataFrame,
                              score_col: str = "score",
                              time_col: str = "open_time",
                              symbol_col: str = "symbol") -> pd.DataFrame:
    """
    Para cada timestamp, calcula z-score e percentil do score cross-section:
      score_cs_mean, score_cs_std, score_z_cs, score_pct_cs (0..1)
    """
    if score_col not in df.columns:
        raise ValueError(f"Coluna {score_col} não encontrada.")
    if time_col not in df.columns or symbol_col not in df.columns:
        raise ValueError("DataFrame precisa de colunas time e symbol.")

    df = df.copy()
    grouped = df.groupby(time_col)[score_col]
    means = grouped.transform("mean")
    stds = grouped.transform("std").replace(0, np.nan)
    df["score_cs_mean"] = means
    df["score_cs_std"] = stds
    df["score_z_cs"] = (df[score_col] - means) / stds

    # Percentil cross-section
    def pct_rank(x):
        return x.rank(pct=True, method="average")
    df["score_pct_cs"] = grouped.transform(pct_rank)

    return df
