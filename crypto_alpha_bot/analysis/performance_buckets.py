import pandas as pd
import numpy as np
from pathlib import Path

from core.labels import add_forward_returns
from core.cross_sectional import add_cross_sectional_stats

HIST_PATH = Path("data/reports/historical_signals.parquet")
ENRICHED_OUT = Path("data/reports/historical_signals_enriched.parquet")


PRICE_CANDIDATES = ["close", "close_price", "c", "last", "price"]


def detect_price_column(df: pd.DataFrame) -> str:
    for c in PRICE_CANDIDATES:
        if c in df.columns:
            return c
    # fallback heurístico
    for c in df.columns:
        if c.lower().startswith("close"):
            return c
    raise ValueError("Nenhuma coluna de preço encontrada para calcular retornos futuros.")


def add_price_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Caso histórico não tenha preço, tenta fazer merge com data/raw/*_1h.parquet.
    """
    if any(c in df.columns for c in PRICE_CANDIDATES):
        return df

    import glob, os
    price_frames = []
    for fp in glob.glob("data/raw/*_1h.parquet"):
        sym = os.path.basename(fp).split("_")[0]
        base = pd.read_parquet(fp)
        if "open_time" in base.columns and "close" in base.columns:
            tmp = base[["open_time", "close"]].copy()
            tmp["symbol"] = sym
            price_frames.append(tmp)
    if not price_frames:
        raise ValueError("Não foi possível adicionar preços (raw vazios).")

    prices = pd.concat(price_frames, ignore_index=True)
    merged = df.merge(prices, on=["symbol", "open_time"], how="left")
    return merged


def compute_buckets(df: pd.DataFrame,
                    score_col: str,
                    horizons=(1, 6, 24),
                    n_buckets: int = 5):
    """
    Calcula métricas de performance por bucket do score_col.
    """
    work = df.copy()
    work = work.dropna(subset=[score_col])

    # Gerar forward returns por símbolo
    price_col = detect_price_column(work)
    work = work.groupby("symbol", group_keys=False).apply(
        lambda g: add_forward_returns(g, price_col=price_col, horizons=horizons)
    )

    # Remover linhas sem preço futuro
    work = work.dropna(subset=[price_col])

    # Buckets globais
    if work[score_col].nunique() < n_buckets:
        n_buckets = work[score_col].nunique()
        print(f"[INFO] Reduzindo n_buckets para {n_buckets} (poucos valores distintos).")

    work["bucket"] = pd.qcut(work[score_col], q=n_buckets, labels=False, duplicates="drop")

    metrics = []
    for b, grp in work.groupby("bucket"):
        row = {
            "bucket": int(b),
            "count": len(grp),
            "score_min": grp[score_col].min(),
            "score_max": grp[score_col].max()
        }
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
                row[f"sharpe_{h}h"] = mean / std if std > 1e-12 else np.nan
        metrics.append(row)

    summary = pd.DataFrame(metrics).sort_values("bucket")
    return summary, work


def main():
    if not HIST_PATH.exists():
        raise FileNotFoundError(f"{HIST_PATH} não existe. Gere com: uv run python main.py --build-hist-signals")

    df = pd.read_parquet(HIST_PATH)

    # Garantir preço
    df = add_price_if_missing(df)

    # Se não existir regime ou score, aborta
    if "score" not in df.columns:
        raise ValueError("Arquivo histórico sem coluna 'score'.")

    # Cross-sectional stats (score_z_cs / score_pct_cs)
    df = add_cross_sectional_stats(df, score_col="score")

    # Score a usar para bucketização
    score_col = "score_z_cs"

    summary, enriched = compute_buckets(df, score_col=score_col, horizons=(1, 6, 24), n_buckets=5)

    # Salvar enriched
    enriched.to_parquet(ENRICHED_OUT, index=False)
    summary_path = ENRICHED_OUT.parent / f"buckets_{score_col}.parquet"
    summary.to_parquet(summary_path, index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\n=== BUCKET PERFORMANCE (cross-sectional) ===")
    print(summary.to_string(index=False))
    print(f"\nSalvo resumo em: {summary_path}")
    print(f"Arquivo enriquecido: {ENRICHED_OUT}")


if __name__ == "__main__":
    main()
