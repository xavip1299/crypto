import pandas as pd
from typing import Dict
from core.ranking import apply_ranking

def generate_historical_signals(price_map: Dict[str, pd.DataFrame],
                                regime_clf,
                                scorer,
                                min_history: int = 50,
                                apply_cross_sectional_rank: bool = False) -> pd.DataFrame:
    """
    Gera sinais históricos para cada símbolo.
    price_map: dict {symbol: DataFrame com colunas incluindo open_time, close e features usados pelo scorer}
    min_history: número mínimo de candles antes de calcular score (evita NaNs em janelas longas)
    apply_cross_sectional_rank: se True, aplica ranking por timestamp (open_time)
    Retorna DataFrame com:
       symbol, open_time, regime, score, momentum, breakout, contrarian, penalty, close
       (e se apply_cross_sectional_rank=True, adiciona colunas do ranking)
    """
    records = []

    for symbol, df in price_map.items():
        if df is None or df.empty:
            continue
        if "open_time" not in df.columns or "close" not in df.columns:
            continue

        df = df.sort_values("open_time").reset_index(drop=True)

        for i in range(len(df)):
            if i < min_history:
                continue

            # janela até i (inclusive) para regime
            hist_slice = df.iloc[: i + 1]
            row = df.iloc[i].to_dict()

            # classificação de regime
            regime = regime_clf.classify(hist_slice)

            # score components
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

    if apply_cross_sectional_rank:
        ranked_parts = []
        for ts, grp in sig_df.groupby("open_time"):
            ranked_parts.append(apply_ranking(grp, use_regime_adjust=True))
        sig_df = pd.concat(ranked_parts, ignore_index=True)

    return sig_df
