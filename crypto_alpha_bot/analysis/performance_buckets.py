import pandas as pd
import numpy as np
from pathlib import Path

from core.labels import add_forward_returns

def compute_buckets(df: pd.DataFrame,
                    score_col: str = "score_z",
                    horizons=(1, 6, 24),
                    n_buckets: int = 5):
    """
    Gera métricas de performance por bucket de score (quintis ou decis se n_buckets=10).
    Requer colunas: symbol, open_time, score_col, close.
    """
    work = df.copy()
    work = work.dropna(subset=[score_col, "close"])
    # Evitar colapso: se score_z for constante, retorna vazio
    if work[score_col].nunique() < 2:
        return pd.DataFrame()

    # Gerar forward returns (log)
    work = work.groupby("symbol", group_keys=False).apply(
        lambda g: add_forward_returns(g, price_col="close", horizons=horizons)
    )

    # Construir buckets globais cross-sectional por tempo (ou simplesmente globais)
    # Simples: buckets globais
    work["bucket"] = pd.qcut(work[score_col], q=n_buckets, labels=False, duplicates="drop")

    metrics = []
    for b, grp in work.groupby("bucket"):
        row = {"bucket": int(b), "count": len(grp), "score_min": grp[score_col].min(), "score_max": grp[score_col].max()}
        for h in horizons:
            col = f"ret_fwd_{h}h"
            sub = grp[col].dropna()
            if sub.empty:
                row[f"mean_{h}h"] = np.nan
                row[f"median_{h}h"] = np.nan
                row[f"hit_{h}h"] = np.nan
                row[f"sharpe_{h}h"] = np.nan
            else:
                mean = sub.mean()
                std = sub.std(ddof=0)
                row[f"mean_{h}h"] = mean
                row[f"median_{h}h"] = sub.median()
                row[f"hit_{h}h"] = (sub > 0).mean()
                row[f"sharpe_{h}h"] = (mean / std) if std > 1e-12 else np.nan
        metrics.append(row)

    res = pd.DataFrame(metrics).sort_values("bucket")
    return res, work


def main():
    reports_path = Path("data/reports")
    hist_path = reports_path / "historical_signals.parquet"
    if not hist_path.exists():
        raise FileNotFoundError(f"Arquivo {hist_path} não encontrado. Gere com: uv run python main.py --build-hist-signals")

    df = pd.read_parquet(hist_path)

    # Escolher score para avaliação: se não tiver score_z, usar score
    score_col = "score_z" if "score_z" in df.columns else "score"
    horizons = (1, 6, 24)
    n_buckets = 5

    summary, enriched = compute_buckets(df, score_col=score_col, horizons=horizons, n_buckets=n_buckets)

    out_summary = reports_path / f"buckets_{score_col}.parquet"
    out_enriched = reports_path / f"historical_signals_enriched.parquet"
    summary.to_parquet(out_summary, index=False)
    enriched.to_parquet(out_enriched, index=False)

    # Print legível
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", None)
    print("\n=== BUCKET PERFORMANCE (log returns) ===")
    print(summary.to_string(index=False))
    print(f"\nSalvo resumo em: {out_summary}")
    print(f"Arquivo enriquecido: {out_enriched}")


if __name__ == "__main__":
    main()
