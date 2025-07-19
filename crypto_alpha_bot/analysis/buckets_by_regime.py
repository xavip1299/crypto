"""
Análise de performance por bucket segmentada por regime de mercado
e estabilidade temporal (primeira metade vs segunda metade).

Pré-requisitos:
  - Arquivo gerado: data/reports/historical_signals_enriched.parquet
    (produzido por performance_buckets.py)
  - Esse arquivo deve conter colunas:
      symbol, open_time, regime, bucket, ret_fwd_1h, ret_fwd_6h, ret_fwd_24h

Uso:
  uv run python analysis/buckets_by_regime.py
"""

from pathlib import Path
import pandas as pd
import numpy as np


ENRICHED_PATH = Path("data/reports/historical_signals_enriched.parquet")


def load_enriched(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo {path} não encontrado.\n"
            "Gere primeiro: uv run python analysis/performance_buckets.py"
        )
    df = pd.read_parquet(path)
    required = {"symbol", "open_time", "regime", "bucket"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas em falta no enriched: {missing}")
    return df


def add_time_fraction(df: pd.DataFrame) -> pd.DataFrame:
    if "open_time" not in df.columns:
        return df
    tmin = df["open_time"].min()
    tmax = df["open_time"].max()
    span = max(1, tmax - tmin)
    df = df.copy()
    df["t_frac"] = (df["open_time"] - tmin) / span
    df["period_half"] = np.where(df["t_frac"] <= 0.5, "early", "late")
    return df


def agg_bucket(grp: pd.DataFrame, horizons=(1, 6, 24)) -> pd.DataFrame:
    rows = []
    for b, sub in grp.groupby("bucket"):
        row = {"bucket": int(b), "count": len(sub)}
        for h in horizons:
            col = f"ret_fwd_{h}h"
            s = sub[col].dropna()
            if s.empty:
                row[f"mean_{h}h"] = np.nan
                row[f"hit_{h}h"] = np.nan
                row[f"sharpe_{h}h"] = np.nan
            else:
                mean = s.mean()
                std = s.std(ddof=0)
                row[f"mean_{h}h"] = mean
                row[f"hit_{h}h"] = (s > 0).mean()
                row[f"sharpe_{h}h"] = mean / std if std > 1e-12 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("bucket")
    return out


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    df = load_enriched(ENRICHED_PATH)
    df = add_time_fraction(df)

    horizons = (1, 6, 24)

    # 1. Visão Global (sanity)
    print_section("VISÃO GLOBAL (Todos os Regimes)")
    global_agg = agg_bucket(df, horizons)
    print(global_agg.to_string(index=False))

    # 2. Por Regime
    print_section("POR REGIME")
    for regime, grp in df.groupby("regime"):
        print(f"\n--- Regime: {regime} ---")
        agg = agg_bucket(grp, horizons)
        print(agg.to_string(index=False))

    # 3. Estabilidade Temporal (Early vs Late)
    if "period_half" in df.columns:
        print_section("ESTABILIDADE TEMPORAL (Metade Inicial vs Final)")
        for half, grp in df.groupby("period_half"):
            print(f"\n--- {half.upper()} ---")
            agg = agg_bucket(grp, horizons)
            print(agg.to_string(index=False))

    # 4. Regime + Period Half combinado (se quiser granular)
    print_section("REGIME x PERIOD_HALF (opcional)")
    combos = []
    for (regime, half), grp in df.groupby(["regime", "period_half"]):
        a = agg_bucket(grp, horizons)
        a.insert(0, "period_half", half)
        a.insert(0, "regime", regime)
        combos.append(a)
    if combos:
        combo_df = pd.concat(combos, ignore_index=True)
        print(combo_df.to_string(index=False))

    # 5. Salvar outputs
    out_dir = ENRICHED_PATH.parent
    (out_dir / "buckets_by_regime.parquet").write_bytes(
        agg_bucket(df, horizons).to_parquet(index=False)
        if hasattr(pd.DataFrame, "to_parquet")
        else b""
    )
    # Também salvar combinado detalhado
    combo_path = out_dir / "buckets_regime_period.parquet"
    try:
        combo_df.to_parquet(combo_path, index=False)
        print(f"\nSalvo resumo combinado em: {combo_path}")
    except Exception as e:
        print(f"[WARN] Falha salvar parquet combinado: {e}")


if __name__ == "__main__":
    main()
