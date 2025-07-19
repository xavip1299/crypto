import pandas as pd
from typing import Dict
from core.regimes import RegimeClassifier
from core.scoring import ScoringEngine

def generate_historical_signals(price_map: Dict[str, pd.DataFrame],
                                regime_clf: RegimeClassifier,
                                scorer: ScoringEngine) -> pd.DataFrame:
    """
    Aplica scoring a cada candle histórico (após features).
    Retorna DataFrame concatenado de sinais com colunas:
      symbol, open_time, regime, score, momentum, breakout, contrarian, penalty
    """
    records = []
    for sym, df in price_map.items():
        if df.empty:
            continue
        regime_series = []
        # Opcional: calcular regime por janela progressiva (mais realista).
        for i in range(len(df)):
            sub = df.iloc[: i + 1]
            regime_series.append(regime_clf.classify(sub))
        last_regime = regime_series  # lista do mesmo tamanho
        for (row, regime) in zip(df.to_dict("records"), last_regime):
            sb = scorer.compute(row, regime)
            records.append({
                "symbol": sym,
                "open_time": row["open_time"],
                "regime": regime,
                "score": sb.total,
                "momentum": sb.momentum,
                "breakout": sb.breakout,
                "contrarian": sb.contrarian,
                "penalty": sb.penalty
            })
    return pd.DataFrame(records)
