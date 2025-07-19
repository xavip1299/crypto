# core/backtest_signals.py

import pandas as pd
from core.ranking import apply_ranking

def generate_historical_signals(price_map, regime_clf, scorer, min_history=50, apply_cross_sectional_rank=False):
    """
    price_map: dict {symbol: DataFrame com colunas incluindo open_time, close, features ...}
    Retorna DataFrame com colunas:
      symbol, open_time, regime, score, momentum, breakout, contrarian, penalty, close
    """
    records = []

    for symbol, df in price_map.items():
        if df.empty or "open_time" not in df.columns or "close" not in df.columns:
            continue

        df = df.sort_values("open_time").reset_index(drop=True)

        for i in range(len(df)):
            # Exigir histórico mínimo para evitar NaNs em médias / std
            if i < min_history:
                continue
            row = df.iloc[i].to_dict()

            regime = regime_clf.classify(df.iloc[:i+1])  # passa só histórico até aqui
            sb = scorer.compute(row, regime)

            records.append({
                "symbol": symbol,
                "open_time": row["open_time"],
                "regime": regime,
                "score": sb.total,
                "momentum": sb.momentum,
                "breakout": sb.breakout,
                "contrarian": sb.contrarian,
                "penalty": sb.penalty,
                "close": row["close"],
            })

    sig_df = pd.DataFrame(records)

    if sig_df.empty:
        return sig_df

    # (Opcional) aplicar ranking cross-sectional por timestamp (open_time)
    if apply_cross_sectional_rank:
        ranked = []
        for ts, grp in sig_df.groupby("open_time"):
            ranked.append(apply_ranking(grp, use_regime_adjust=True))
        sig_df = pd.concat(ranked, ignore_index=True)

    return sig_df
