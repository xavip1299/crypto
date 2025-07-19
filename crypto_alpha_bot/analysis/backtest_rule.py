from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

from core.cross_sectional import add_cross_sectional_stats

ENRICHED_PATH = Path("data/reports/historical_signals_enriched.parquet")

@dataclass
class Trade:
    symbol: str
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    ret_pct_gross: float
    ret_pct_net: float
    hold_hours: int
    reason: str
    score_z_entry: float
    bucket_entry: int
    funding_z_entry: float | None
    oi_delta_entry: float | None

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado.")
    df = pd.read_parquet(path)
    needed = {"symbol","open_time","close","score"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"Faltam colunas: {miss}")
    return df.sort_values(["open_time","symbol"]).reset_index(drop=True)

def compute_bucket(df: pd.DataFrame, score_col: str, n_buckets: int):
    df = df.copy()
    df["bucket"] = pd.qcut(df[score_col], q=n_buckets, labels=False, duplicates="drop")
    return df

def satisfies_filters(row,
                      allowed_regimes,
                      min_funding_z,
                      min_oi_delta,
                      min_score_pct,
                      require_score_z_nonneg):
    if allowed_regimes and str(row.regime) not in allowed_regimes:
        return False
    if min_funding_z is not None and not np.isnan(row.get("funding_z", np.nan)):
        if row.funding_z < min_funding_z:
            return False
    if min_oi_delta is not None and not np.isnan(row.get("oi_delta_pct", np.nan)):
        if row.oi_delta_pct < min_oi_delta:
            return False
    if min_score_pct is not None and not np.isnan(row.get("score_pct_cs", np.nan)):
        if row.score_pct_cs < min_score_pct:
            return False
    if require_score_z_nonneg and not np.isnan(row.get("score_z_cs", np.nan)):
        if row.score_z_cs < 0:
            return False
    return True

def backtest(df: pd.DataFrame,
             mode: str,
             n_buckets: int,
             entry_bucket: int | None,
             z_entry: float,
             tp_mult: float,
             sl_mult: float,
             max_hold_hours: int,
             atr_window: int,
             cost_perc: float,
             slip_perc: float,
             filters: dict) -> list[Trade]:

    # ATR percent (close-based)
    df = df.copy()
    if {"high","low","close"}.issubset(df.columns):
        tr = df["high"] - df["low"]
        df["atr_pct"] = tr.rolling(atr_window).mean() / df["close"]
    else:
        df["atr_pct"] = df["close"].pct_change().abs().rolling(atr_window).mean()

    # Garantir cross-sectional stats
    df = add_cross_sectional_stats(df, score_col="score", time_col="open_time")

    score_for_buckets = "score_z_cs"
    df = compute_bucket(df, score_for_buckets, n_buckets)
    if mode == "bucket" and entry_bucket is None:
        entry_bucket = int(df["bucket"].max())

    trades: list[Trade] = []
    open_pos: dict[str, Trade] = {}

    for symbol, g in df.groupby("symbol"):
        g = g.sort_values("open_time").reset_index(drop=True)

        for i in range(len(g)-1):
            row = g.iloc[i]
            nxt = g.iloc[i+1]

            # Close existing positions first
            to_close = []
            for sym, pos in list(open_pos.items()):
                if sym != symbol:
                    continue
                current_price = row.close
                gross_ret = (current_price / pos.entry_price) - 1.0
                hold_hours = int((row.open_time - pos.entry_time)//3_600_000)

                # Dynamic TP/SL
                atr = row.atr_pct if not np.isnan(row.atr_pct) else 0.01
                tp = tp_mult * atr
                sl = sl_mult * atr

                reason = None
                if gross_ret >= tp:
                    reason = "TP"
                elif gross_ret <= -sl:
                    reason = "SL"
                elif hold_hours >= max_hold_hours:
                    reason = "TIME"

                if reason:
                    net_ret = gross_ret - cost_perc - slip_perc  # saída custos
                    pos.exit_time = int(row.open_time)
                    pos.exit_price = current_price
                    pos.ret_pct_gross = gross_ret
                    pos.ret_pct_net = net_ret
                    pos.hold_hours = hold_hours
                    pos.reason = reason
                    trades.append(pos)
                    to_close.append(sym)
            for sym in to_close:
                open_pos.pop(sym, None)

            # Entry
            if symbol not in open_pos:
                if not satisfies_filters(row,
                                         filters.get("allowed_regimes"),
                                         filters.get("min_funding_z"),
                                         filters.get("min_oi_delta"),
                                         filters.get("min_score_pct"),
                                         filters.get("require_score_z_nonneg")):
                    continue
                cond = False
                if mode == "bucket":
                    cond = (row.bucket == entry_bucket)
                else:
                    cond = (row.score_z_cs >= z_entry)

                if cond and not np.isnan(row.close):
                    entry_price = nxt.close * (1 + slip_perc)  # slip na entrada
                    gross_cost = cost_perc  # custo entrada (irá descontar na saída também)
                    trade = Trade(
                        symbol=symbol,
                        entry_time=int(nxt.open_time),
                        exit_time=0,
                        entry_price=entry_price,
                        exit_price=0.0,
                        ret_pct_gross=0.0,
                        ret_pct_net=-gross_cost,  # custo inicial
                        hold_hours=0,
                        reason="",
                        score_z_entry=row.score_z_cs if "score_z_cs" in row else np.nan,
                        bucket_entry=int(row.bucket),
                        funding_z_entry=row.get("funding_z", np.nan),
                        oi_delta_entry=row.get("oi_delta_pct", np.nan)
                    )
                    open_pos[symbol] = trade

        # Force close last
        last = g.iloc[-1]
        for sym, pos in list(open_pos.items()):
            if sym == symbol:
                current_price = last.close
                gross_ret = (current_price / pos.entry_price) - 1.0
                hold_hours = int((last.open_time - pos.entry_time)//3_600_000)
                net_ret = gross_ret - cost_perc - slip_perc
                pos.exit_time = int(last.open_time)
                pos.exit_price = current_price
                pos.ret_pct_gross = gross_ret
                pos.ret_pct_net += net_ret  # inclui custo inicial já subtraído
                pos.hold_hours = hold_hours
                pos.reason = "EOL"
                trades.append(pos)
                open_pos.pop(sym, None)

    return trades

def summarize(trades: list[Trade]):
    if not trades:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame([t.__dict__ for t in trades])
    df = df.sort_values("exit_time")
    df["equity"] = (1 + df.ret_pct_net).cumprod()
    roll_max = df.equity.cummax()
    dd = df.equity / roll_max - 1
    summary = {
        "trades": len(df),
        "win_rate": (df.ret_pct_net > 0).mean(),
        "avg_ret_net": df.ret_pct_net.mean(),
        "median_ret_net": df.ret_pct_net.median(),
        "std_ret_net": df.ret_pct_net.std(ddof=0),
        "sharpe_net": df.ret_pct_net.mean()/df.ret_pct_net.std(ddof=0) if df.ret_pct_net.std(ddof=0)>1e-12 else np.nan,
        "final_equity": df.equity.iloc[-1],
        "max_dd_equity": dd.min(),
        "best_trade": df.ret_pct_net.max(),
        "worst_trade": df.ret_pct_net.min(),
        "avg_hold_h": df.hold_hours.mean()
    }
    return pd.DataFrame([summary]), df

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bucket","zscore"], default="bucket")
    ap.add_argument("--buckets", type=int, default=5)
    ap.add_argument("--entry-bucket", type=int, default=None)
    ap.add_argument("--z-entry", type=float, default=0.5)
    ap.add_argument("--tp-mult", type=float, default=2.5, help="Multiplicador ATR para TP")
    ap.add_argument("--sl-mult", type=float, default=1.2, help="Multiplicador ATR para SL")
    ap.add_argument("--atr-window", type=int, default=14)
    ap.add_argument("--max-hold", type=int, default=24)
    ap.add_argument("--cost-bps", type=float, default=7, help="Custo (ida ou venida) total em basis points (somado nos dois lados).")
    ap.add_argument("--slip-bps", type=float, default=5, help="Slippage de execução em bps (aplicado entrada e saída).")
    # Filters
    ap.add_argument("--regimes", type=str, default="Vol_Expansao,Vol_Compressa,Tendencia_Alta")
    ap.add_argument("--min-funding-z", type=float, default=-0.5)
    ap.add_argument("--min-oi-delta", type=float, default=None)
    ap.add_argument("--min-score-pct", type=float, default=0.8)
    ap.add_argument("--require-score-z-nonneg", action="store_true")
    ap.add_argument("--export-trades", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    df = load_data(ENRICHED_PATH)

    # Filtros dicionário
    filters = {
        "allowed_regimes": [r.strip() for r in args.regimes.split(",") if r.strip()],
        "min_funding_z": args.min_funding_z,
        "min_oi_delta": args.min_oi_delta,
        "min_score_pct": args.min_score_pct,
        "require_score_z_nonneg": args.require_score_z_nonneg
    }

    trades = backtest(
        df,
        mode=args.mode,
        n_buckets=args.buckets,
        entry_bucket=args.entry_bucket,
        z_entry=args.z_entry,
        tp_mult=args.tp_mult,
        sl_mult=args.sl_mult,
        max_hold_hours=args.max_hold,
        atr_window=args.atr_window,
        cost_perc=args.cost_bps / 10000.0,
        slip_perc=args.slip_bps / 10000.0,
        filters=filters
    )

    summary, trades_df = summarize(trades)
    if summary.empty:
        print("Sem trades após filtros.")
        return

    print("\n=== BACKTEST SUMMARY (Filtros + Custos) ===")
    print(summary.to_string(index=False))

    print("\n=== PRIMEIRAS 5 TRADES ===")
    print(trades_df.head(5).to_string(index=False))
    print("\n=== ÚLTIMAS 5 TRADES ===")
    print(trades_df.tail(5).to_string(index=False))

    out_dir = Path("data/reports")
    out_dir.mkdir(exist_ok=True, parents=True)
    summary.to_parquet(out_dir / "backtest_summary_filtered.parquet", index=False)
    if args.export_trades:
        trades_df.to_parquet(out_dir / "backtest_trades_filtered.parquet", index=False)
        print(f"\nTrades salvas em: {out_dir / 'backtest_trades_filtered.parquet'}")

if __name__ == "__main__":
    main()
