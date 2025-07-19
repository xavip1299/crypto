from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from core.cross_sectional import add_cross_sectional_stats

ENRICHED_PATH = Path("data/reports/historical_signals_enriched.parquet")


# ============================= DATA CLASSES ============================= #

@dataclass
class Position:
    symbol: str
    entry_time: int
    entry_price: float
    size: float
    peak_ret: float
    score_z_entry: float
    bucket_entry: int
    funding_z_entry: float | None
    oi_delta_entry: float | None
    partial_done: bool = False
    ret_open: float = 0.0


@dataclass
class ClosedTrade:
    symbol: str
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    gross_ret: float
    net_ret: float
    hold_hours: int
    reason: str
    score_z_entry: float
    bucket_entry: int
    funding_z_entry: float | None
    oi_delta_entry: float | None


# ============================= HELPERS ============================= #

def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado.")
    df = pd.read_parquet(path)
    need = {"symbol", "open_time", "close", "score"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Faltam colunas necessárias: {miss}")
    return df.sort_values(["open_time", "symbol"]).reset_index(drop=True)


def add_cs(df: pd.DataFrame):
    return add_cross_sectional_stats(df, score_col="score", time_col="open_time")


def compute_bucket(df: pd.DataFrame, score_col="score_z_cs", q=5):
    df = df.copy()
    df["bucket"] = pd.qcut(df[score_col], q=q, labels=False, duplicates="drop")
    return df


def satisfies(row, flt):
    if flt["regimes"] and str(row.regime) not in flt["regimes"]:
        return False
    if flt["min_score_pct"] is not None and row.get("score_pct_cs", np.nan) < flt["min_score_pct"]:
        return False
    if flt["req_score_nonneg"] and row.get("score_z_cs", -1) < 0:
        return False
    if flt["min_funding_z"] is not None and not np.isnan(row.get("funding_z", np.nan)):
        if row.funding_z < flt["min_funding_z"]:
            return False
    if flt["min_oi_delta"] is not None and not np.isnan(row.get("oi_delta_pct", np.nan)):
        if row.oi_delta_pct < flt["min_oi_delta"]:
            return False
    if flt["min_vol_pct"] is not None and not np.isnan(row.get("vol_pct", np.nan)):
        if row.vol_pct < flt["min_vol_pct"]:
            return False
    return True


# ============================= CORE BACKTEST ============================= #

def backtest(df: pd.DataFrame, params, flt):
    df = df.copy()

    # ATR / Vol proxy
    if {"high", "low", "close"}.issubset(df.columns):
        tr = (df["high"] - df["low"]).abs()
        df["atr_pct"] = tr.rolling(params.atr_window).mean() / df["close"]
    else:
        df["atr_pct"] = df["close"].pct_change().abs().rolling(params.atr_window).mean()

    # Cross-sectional stats e buckets
    df = add_cs(df)
    df = compute_bucket(df, "score_z_cs", q=params.buckets)

    if params.entry_bucket is None:
        params.entry_bucket = int(df["bucket"].max())

    positions: dict[str, Position] = {}
    closed: list[ClosedTrade] = []

    for symbol, g in df.groupby("symbol"):
        g = g.sort_values("open_time").reset_index(drop=True)
        for i in range(len(g) - 1):
            row = g.iloc[i]
            nxt = g.iloc[i + 1]  # Execução próxima barra
            atr = row.atr_pct if not np.isnan(row.atr_pct) and row.atr_pct > 0 else 0.01

            # 1. Atualizar posições abertas
            remove_syms = []
            for sym, pos in positions.items():
                if sym != symbol:
                    continue
                current_ret = (row.close / pos.entry_price) - 1.0
                pos.peak_ret = max(pos.peak_ret, current_ret)
                pos.ret_open = current_ret
                hold_h = int((row.open_time - pos.entry_time) // 3_600_000)

                # Parâmetros dinâmicos
                tp2 = params.tp2_mult * atr
                tp1 = params.tp1_mult * atr
                sl = params.sl_mult * atr

                # Trailing só começa após ganho mínimo (>= 50% do nível de TP1)
                trail_min_gain = 0.5 * tp1
                trail_trigger = (
                    pos.peak_ret >= trail_min_gain and
                    (pos.peak_ret - current_ret) >= params.trailing_decay * max(pos.peak_ret, 1e-9)
                )

                # Partial exit
                if (not pos.partial_done) and current_ret >= tp1:
                    gross_half = current_ret
                    net_half = gross_half - params.cost_perc - params.slip_perc
                    closed.append(
                        ClosedTrade(
                            symbol=symbol,
                            entry_time=pos.entry_time,
                            exit_time=int(row.open_time),
                            entry_price=pos.entry_price,
                            exit_price=row.close,
                            gross_ret=gross_half * 0.5,
                            net_ret=net_half * 0.5,
                            hold_hours=hold_h,
                            reason="PARTIAL",
                            score_z_entry=pos.score_z_entry,
                            bucket_entry=pos.bucket_entry,
                            funding_z_entry=pos.funding_z_entry,
                            oi_delta_entry=pos.oi_delta_entry,
                        )
                    )
                    pos.size *= 0.5
                    pos.partial_done = True

                # Early exit conservador
                early_exit = (
                    params.early_exit and
                    hold_h >= 2 and
                    current_ret < 0 and
                    row.get("score_z_cs", 0) < -0.5
                )

                hit_tp2 = current_ret >= tp2
                hit_sl = current_ret <= -sl
                time_exit = hold_h >= params.max_hold

                if hit_tp2 or hit_sl or time_exit or trail_trigger or early_exit:
                    reason = (
                        "TP2" if hit_tp2 else
                        "SL" if hit_sl else
                        "TIME" if time_exit else
                        "TRAIL" if trail_trigger else
                        "EARLY"
                    )
                    gross = (row.close / pos.entry_price) - 1.0
                    net = gross - params.cost_perc - params.slip_perc
                    closed.append(
                        ClosedTrade(
                            symbol=symbol,
                            entry_time=pos.entry_time,
                            exit_time=int(row.open_time),
                            entry_price=pos.entry_price,
                            exit_price=row.close,
                            gross_ret=gross * pos.size,
                            net_ret=net * pos.size,
                            hold_hours=hold_h,
                            reason=reason,
                            score_z_entry=pos.score_z_entry,
                            bucket_entry=pos.bucket_entry,
                            funding_z_entry=pos.funding_z_entry,
                            oi_delta_entry=pos.oi_delta_entry,
                        )
                    )
                    remove_syms.append(sym)

            for sym in remove_syms:
                positions.pop(sym, None)

            # 2. Entrada
            if symbol not in positions:
                if not satisfies(row, flt):
                    continue
                cond = (
                    (params.mode == "bucket" and row.bucket == params.entry_bucket) or
                    (params.mode == "zscore" and row.score_z_cs >= params.z_entry)
                )
                if cond:
                    entry_price = nxt.close * (1 + params.slip_perc)
                    positions[symbol] = Position(
                        symbol=symbol,
                        entry_time=int(nxt.open_time),
                        entry_price=entry_price,
                        size=1.0,
                        peak_ret=0.0,
                        score_z_entry=row.score_z_cs,
                        bucket_entry=int(row.bucket),
                        funding_z_entry=row.get("funding_z", np.nan),
                        oi_delta_entry=row.get("oi_delta_pct", np.nan),
                    )

        # 3. Fechar remanescentes na última barra
        last = g.iloc[-1]
        for sym, pos in list(positions.items()):
            if sym == symbol:
                gross = (last.close / pos.entry_price) - 1.0
                net = gross - params.cost_perc - params.slip_perc
                hold_h = int((last.open_time - pos.entry_time) // 3_600_000)
                closed.append(
                    ClosedTrade(
                        symbol=symbol,
                        entry_time=pos.entry_time,
                        exit_time=int(last.open_time),
                        entry_price=pos.entry_price,
                        exit_price=last.close,
                        gross_ret=gross * pos.size,
                        net_ret=net * pos.size,
                        hold_hours=hold_h,
                        reason="EOL",
                        score_z_entry=pos.score_z_entry,
                        bucket_entry=pos.bucket_entry,
                        funding_z_entry=pos.funding_z_entry,
                        oi_delta_entry=pos.oi_delta_entry,
                    )
                )
                positions.pop(sym, None)

    return closed


# ============================= SUMMARIZE ============================= #

def summarize(trades: list[ClosedTrade]):
    if not trades:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame([t.__dict__ for t in trades]).sort_values("exit_time")
    df["equity"] = (1 + df.net_ret).cumprod()
    roll_max = df.equity.cummax()
    dd = df.equity / roll_max - 1

    summary = {
        "trades": len(df),
        "win_rate": (df.net_ret > 0).mean(),
        "avg_net_ret": df.net_ret.mean(),
        "median_net_ret": df.net_ret.median(),
        "std_net_ret": df.net_ret.std(ddof=0),
        "sharpe_net": df.net_ret.mean() / df.net_ret.std(ddof=0) if df.net_ret.std(ddof=0) > 1e-12 else np.nan,
        "final_equity": df.equity.iloc[-1],
        "max_dd": dd.min(),
        "best_trade": df.net_ret.max(),
        "worst_trade": df.net_ret.min(),
        "avg_hold_h": df.hold_hours.mean(),
        "partial_trades": (df.reason == "PARTIAL").sum(),
        "tp2_trades": (df.reason == "TP2").sum(),
        "trail_trades": (df.reason == "TRAIL").sum(),
        "early_exits": (df.reason == "EARLY").sum(),
    }
    return pd.DataFrame([summary]), df


# ============================= ARGS & MAIN ============================= #

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bucket", "zscore"], default="bucket")
    ap.add_argument("--buckets", type=int, default=5)
    ap.add_argument("--entry-bucket", type=int, default=None)
    ap.add_argument("--z-entry", type=float, default=0.8)
    ap.add_argument("--max-hold", type=int, default=24)

    ap.add_argument("--tp1-mult", type=float, default=1.2)
    ap.add_argument("--tp2-mult", type=float, default=2.5)
    ap.add_argument("--sl-mult", type=float, default=0.9)
    ap.add_argument("--trailing-decay", type=float, default=0.5,
                    help="Fechar se devolve >= decay * peak_ret (ex: 0.5 => devolve metade).")
    ap.add_argument("--early-exit", action="store_true", help="Ativa early-exit conservador (score_z_cs < -0.5, ret<0, hold>=2h).")

    ap.add_argument("--atr-window", type=int, default=14)

    # Filtros
    ap.add_argument("--min-score-pct", type=float, default=0.9)
    ap.add_argument("--regimes", type=str, default="Vol_Expansao,Vol_Compressa,Tendencia_Alta")
    ap.add_argument("--min-funding-z", type=float, default=-0.5)
    ap.add_argument("--min-oi-delta", type=float, default=None)
    ap.add_argument("--min-vol-pct", type=float, default=None)
    ap.add_argument("--req-score-nonneg", action="store_true")

    # Custos
    ap.add_argument("--cost-bps", type=float, default=10)
    ap.add_argument("--slip-bps", type=float, default=7)

    ap.add_argument("--export-trades", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    df = load_data(ENRICHED_PATH)

    flt = {
        "regimes": [r.strip() for r in args.regimes.split(",") if r.strip()],
        "min_score_pct": args.min_score_pct,
        "req_score_nonneg": args.req_score_nonneg,
        "min_funding_z": args.min_funding_z,
        "min_oi_delta": args.min_oi_delta,
        "min_vol_pct": args.min_vol_pct,
    }

    class Params: ...
    params = Params()
    params.mode = args.mode
    params.buckets = args.buckets
    params.entry_bucket = args.entry_bucket
    params.z_entry = args.z_entry
    params.max_hold = args.max_hold
    params.tp1_mult = args.tp1_mult
    params.tp2_mult = args.tp2_mult
    params.sl_mult = args.sl_mult
    params.trailing_decay = args.trailing_decay
    params.early_exit = args.early_exit
    params.atr_window = args.atr_window
    params.cost_perc = args.cost_bps / 10000.0
    params.slip_perc = args.slip_bps / 10000.0

    trades = backtest(df, params, flt)
    summary, trades_df = summarize(trades)

    if summary.empty:
        print("Sem trades.")
        return

    print("\n=== BACKTEST SUMMARY (ADV) ===")
    print(summary.to_string(index=False))

    print("\n=== DISTRIBUIÇÃO REASONS ===")
    print(trades_df.reason.value_counts())

    print("\n=== ÚLTIMAS 8 TRADES ===")
    print(trades_df.tail(8).to_string(index=False))

    out_dir = Path("data/reports")
    out_dir.mkdir(exist_ok=True, parents=True)
    summary.to_parquet(out_dir / "backtest_summary_adv.parquet", index=False)
    if args.export_trades:
        trades_df.to_parquet(out_dir / "backtest_trades_adv.parquet", index=False)
        print(f"\nTrades salvas em: {out_dir / 'backtest_trades_adv.parquet'}")


if __name__ == "__main__":
    main()
