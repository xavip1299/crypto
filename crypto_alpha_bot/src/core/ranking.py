import pandas as pd
from core.regimes import Regime

REGIME_FACTORS = {
    Regime.TEND_ALTA: 1.0,
    Regime.TEND_BAIXA: 1.0,
    Regime.VOL_COMPRESSA: 1.10,
    Regime.VOL_EXPANSAO: 0.95,  # <-- corrigido
    Regime.CHOP: 0.90,
}

def apply_ranking(df: pd.DataFrame, use_regime_adjust: bool = True) -> pd.DataFrame:
    """
    Adiciona colunas:
      - regime_factor
      - score_adj
      - score_rank (1 = melhor)
      - score_pct
      - score_z
    """
    if df.empty or "score" not in df.columns:
        return df.copy()

    work = df.copy()

    if use_regime_adjust and "regime" in work.columns:
        def _factor(row):
            return REGIME_FACTORS.get(row.regime, 1.0)
        work["regime_factor"] = work.apply(_factor, axis=1)
    else:
        work["regime_factor"] = 1.0

    work["score_adj"] = work["score"] * work["regime_factor"]
    work = work.sort_values("score_adj", ascending=False, kind="mergesort")
    work["score_rank"] = range(1, len(work) + 1)

    n = len(work)
    work["score_pct"] = 1.0 if n <= 1 else 1.0 - (work["score_rank"] - 1) / (n - 1)

    mean = work["score_adj"].mean()
    std = work["score_adj"].std()
    work["score_z"] = 0.0 if std < 1e-12 else (work["score_adj"] - mean) / std

    return work.sort_values("score_rank")
