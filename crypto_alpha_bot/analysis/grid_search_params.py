import itertools
from pathlib import Path
import pandas as pd
import numpy as np
from subprocess import run, PIPE
import json
import os

# Este script chama internamente o backtest simples (não o advanced) para rapidez.
# Ajusta se quiseres usar o advanced.

BACKTEST_CMD = "uv run python analysis/backtest_rule.py"
REPORTS_DIR = Path("data/reports")

param_space = {
    "min_score_pct": [0.80, 0.85, 0.90, 0.95],
    "tp_mult": [2.0, 2.5, 3.0],
    "sl_mult": [0.8, 1.0, 1.2],
    "max_hold": [12, 24]
}

def parse_summary(parquet_path: Path):
    if not parquet_path.exists():
        return None
    df = pd.read_parquet(parquet_path)
    # Move file to unique name so next run overwrites fresh
    return df.iloc[-1].to_dict()

def main():
    results = []
    combos = list(itertools.product(*param_space.values()))
    keys = list(param_space.keys())

    for combo in combos:
        params = dict(zip(keys, combo))
        print(f"Rodando combo: {params}")
        cmd = (f"{BACKTEST_CMD} "
               f"--min-score-pct {params['min_score_pct']} "
               f"--tp-mult {params['tp_mult']} "
               f"--sl-mult {params['sl_mult']} "
               f"--max-hold {params['max_hold']} "
               f"--require-score-z-nonneg")
        proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        # Ignorar stdout; ler summary parquet
        summary_path = REPORTS_DIR / "backtest_summary.parquet"
        summary = parse_summary(summary_path)
        if summary:
            summary.update(params)
            results.append(summary)

    if results:
        out_df = pd.DataFrame(results)
        out_df = out_df.sort_values("sharpe_net", ascending=False)
        out_path = REPORTS_DIR / "grid_results.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"\nTop 10 combinações por sharpe_net:")
        print(out_df.head(10).to_string(index=False))
        print(f"\nResultados completos em: {out_path}")
    else:
        print("Nenhum resultado coletado (verifique erros).")

if __name__ == "__main__":
    main()
