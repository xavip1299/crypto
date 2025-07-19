import pandas as pd
import numpy as np

def add_extended_micro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona features leves:
      - ret_log_{1h,6h,24h}
      - mom_multi (combinação ponderada)
      - oi_divergence = d(oi%) - d(price%)
      - funding_dev = funding - média(24)
      - funding_z_24 (z-score intrínseco de 24h)
      - vol_pct (ATR simples relativo ao close)
    Requer colunas (se existirem): close, open_interest_usd, funding
    """
    if df.empty:
        return df
    out = df.copy()

    if "close" in out.columns:
        close = out["close"].astype(float)
        out["ret_log_1h"] = np.log(close / close.shift(1))
        out["ret_log_6h"] = np.log(close / close.shift(6))
        out["ret_log_24h"] = np.log(close / close.shift(24))
        # momentum combinado
        out["mom_multi"] = (0.5 * out["ret_log_1h"].fillna(0) +
                            0.3 * out["ret_log_6h"].fillna(0) +
                            0.2 * out["ret_log_24h"].fillna(0))
        # volatilidade (ATR simplificada: high-low ou retorno abs)
        if {"high","low"}.issubset(out.columns):
            tr = (out["high"] - out["low"]).abs()
        else:
            tr = close.pct_change().abs()
        out["vol_pct"] = (tr.rolling(14).mean() / close).replace([np.inf,-np.inf], np.nan)

    if {"open_interest_usd","close"}.issubset(out.columns):
        price_ret = out["close"].pct_change()
        oi_ret = out["open_interest_usd"].pct_change()
        out["oi_divergence"] = oi_ret - price_ret

    if "funding" in out.columns:
        f = out["funding"].astype(float)
        ma24 = f.rolling(24).mean()
        std24 = f.rolling(24).std()
        out["funding_dev"] = f - ma24
        out["funding_z_24"] = (f - ma24) / std24.replace(0, np.nan)

    return out
