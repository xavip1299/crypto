import pandas as pd
import numpy as np

def add_forward_returns(df: pd.DataFrame,
                        price_col: str = "close",
                        horizons: tuple[int, ...] = (1, 6, 24),
                        time_col: str = "open_time",
                        sort: bool = True) -> pd.DataFrame:
    """
    Adiciona colunas de retorno log futuro para cada horizonte horário especificado.
    Requer que os candles sejam contínuos por intervalo fixo (ex. 1h).
    Retorno log: ln(P_{t+h} / P_t)
    """
    if df.empty or price_col not in df.columns:
        return df
    work = df.copy()
    if sort:
        work = work.sort_values(time_col)
    prices = work[price_col].astype(float)
    for h in horizons:
        work[f"ret_fwd_{h}h"] = np.log(prices.shift(-h) / prices)
    return work
