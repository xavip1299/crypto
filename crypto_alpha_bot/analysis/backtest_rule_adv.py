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
    sym_z_entry: float | None
    partial_done: bool = False
    realized_partial_ret: float = 0.0
    peak_price: float = 0.0


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
    sym_z_entry: float | None
    partial_hit: bool
    tp2_hit: bool
    trail_used: bool


@dataclass
class ArmedSignal:
    """Sinal aguardando confirmação do próximo candle."""
    symbol: str
    arm_time: int          # open_time onde sinal surgiu
    planned_entry_time: int  # open_time esperado para entrada (próximo candle)
    score_z_cs: float
    bucket: int
    funding_z: float | None
    oi_delta: float | None
    sym_z: float | None
    entry_ref_close: float  # close do candle de arm
    filters_snapshot: dict  # snapshot de métricas usadas
    score_pct_cs: float | None


# ============================= HELPERS ============================= #

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado.")
    df = pd.read_parquet(path)
    need = {"symbol", "open_time", "close", "score"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Faltam colunas: {miss}")
    return df.sort_values(["open_time", "symbol"]).reset_index(drop=True)


def add_cs(df: pd.DataFrame):
    return add_cross_sectional_stats(df, score_col="score", time_col="open_time")


def compute_bucket(df: pd.DataFrame, score_col="score_z_cs", q=5):
    df = df.copy()
    df["bucket"] = pd.qcut(df[score_col], q=q, labels=False, duplicates="drop")
    return df


def satisfies_basic(row, flt):
    if flt["regimes"] and str(row.regime) not in flt["regimes"]:
        return False
    if flt["exclude_regimes"] and str(row.regime) in flt["exclude_regimes"]:
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


def satisfies_factor_filters(row, params):
    # momentum / breakout / contrarian thresholds if columns exist
    if params.momentum_col in row.index and params.min_momentum is not None:
        if pd.notna(row[params.momentum_col]) and row[params.momentum_col] < params.min_momentum:
            return False
    if params.breakout_col in row.index and params.min_breakout is not None:
        if pd.notna(row[params.breakout_col]) and row[params.breakout_col] < params.min_breakout:
            return False
    if params.contrarian_col in row.index and params.min_contrarian is not None:
        if pd.notna(row[params.contrarian_col]) and row[params.contrarian_col] < params.min_contrarian:
            return False
    return True


def choose_targets(params, atr):
    if params.fixed_tp1_pct is not None:
        tp1 = params.fixed_tp1_pct
    else:
        tp1 = params.tp1_mult * atr
    if params.fixed_tp2_pct is not None:
        tp2 = params.fixed_tp2_pct
    else:
        tp2 = params.tp2_mult * atr
    if params.fixed_sl_pct is not None:
        sl = params.fixed_sl_pct
    else:
        sl = params.sl_mult * atr
    return tp1, tp2, sl


# ============================= CORE BACKTEST ============================= #

def backtest(df: pd.DataFrame, params, flt):
    df = df.copy()

    # ATR proxy
    if {"high", "low", "close"}.issubset(df.columns):
        tr = (df["high"] - df["low"]).abs()
        df["atr_pct"] = tr.rolling(params.atr_window).mean() / df["close"]
    else:
        df["atr_pct"] = df["close"].pct_change().abs().rolling(params.atr_window).mean()

    df = add_cs(df)
    df = compute_bucket(df, "score_z_cs", q=params.buckets)

    if params.double_confirm:
        df["score_z_cs_prev"] = df.groupby("symbol")["score_z_cs"].shift(1)
        if "score_z_sym" in df.columns:
            df["score_z_sym_prev"] = df.groupby("symbol")["score_z_sym"].shift(1)

    if params.entry_bucket is None:
        params.entry_bucket = int(df["bucket"].max())

    positions: dict[str, Position] = {}
    closed: list[ClosedTrade] = []
    last_exit_time: dict[str, int] = {}
    entries_per_hour: dict[int, int] = {}
    armed_signals: dict[str, ArmedSignal] = {}

    # Global risk-off
    def recent_losses_exceeded():
        if params.max_losses_window is None or params.max_losses_allowed is None:
            return False
        recent = trades_global[-params.max_losses_window:] if len(trades_global) else []
        losses = [t for t in recent if t.net_ret <= 0]
        return len(losses) > params.max_losses_allowed

    trades_global: list[ClosedTrade] = []
    risk_off_until_time = None

    # Map last losing entry time per symbol
    last_loss_exit_time: dict[str, int] = {}

    for symbol, g in df.groupby("symbol"):
        g = g.sort_values("open_time").reset_index(drop=True)

        for i in range(len(g) - 1):
            row = g.iloc[i]
            nxt = g.iloc[i + 1]
            atr = row.atr_pct if not np.isnan(row.atr_pct) and row.atr_pct > 0 else 0.01
            tp1, tp2, sl = choose_targets(params, atr)

            # =========== Atualizar posição aberta ===========
            if symbol in positions:
                pos = positions[symbol]
                current_ret = (row.close / pos.entry_price) - 1.0
                pos.peak_ret = max(pos.peak_ret, current_ret)
                pos.peak_price = max(pos.peak_price, row.close)
                hold_h = int((row.open_time - pos.entry_time) // 3_600_000)

                hit_tp2 = current_ret >= tp2
                hit_sl = current_ret <= -sl
                hit_partial = (not pos.partial_done) and (current_ret >= tp1) and (not hit_tp2)

                trail_allowed = True
                if params.trail_requires_partial and not pos.partial_done:
                    trail_allowed = False
                gate_ret = tp1 + params.trail_after_fraction * (tp2 - tp1) if params.trail_after_fraction is not None else tp1
                if current_ret < gate_ret:
                    trail_allowed = False

                rel_dd = pos.peak_ret - current_ret
                dd_rel_cond = rel_dd >= params.trailing_decay * max(pos.peak_ret, 1e-9)
                dd_abs_cond = (params.trail_dd_pct is not None) and (rel_dd >= params.trail_dd_pct)
                trail_trigger = trail_allowed and (dd_rel_cond or dd_abs_cond)

                time_exit = hold_h >= params.max_hold

                early_exit_cond = False
                if params.early_exit and hold_h >= params.early_min_hold and current_ret < 0:
                    metric = row.get("score_z_sym", np.nan) if (params.use_symbol_z_for_early and "score_z_sym" in row) else row.get("score_z_cs", 0)
                    early_exit_cond = metric < params.early_z_threshold

                # Partial interno
                if hit_partial:
                    frac = params.partial_fraction
                    realized_part = current_ret * frac
                    net_part = realized_part - (params.cost_perc + params.slip_perc) * frac
                    pos.realized_partial_ret += net_part
                    pos.size *= (1 - frac)
                    pos.partial_done = True

                reason = None
                tp2_hit_flag = False
                trail_flag = False

                if hit_tp2:
                    reason = "TP2"; tp2_hit_flag = True
                elif hit_sl:
                    reason = "SL"
                elif trail_trigger:
                    reason = "TRAIL"; trail_flag = True
                elif time_exit:
                    reason = "TIME"
                elif early_exit_cond:
                    reason = "EARLY"

                if reason:
                    gross_remaining = (row.close / pos.entry_price) - 1.0
                    net_remaining = gross_remaining - (params.cost_perc + params.slip_perc)
                    total_net = pos.realized_partial_ret + net_remaining * pos.size
                    total_gross = pos.realized_partial_ret + gross_remaining * pos.size
                    trade = ClosedTrade(
                        symbol=symbol,
                        entry_time=pos.entry_time,
                        exit_time=int(row.open_time),
                        entry_price=pos.entry_price,
                        exit_price=row.close,
                        gross_ret=total_gross,
                        net_ret=total_net,
                        hold_hours=hold_h,
                        reason=reason,
                        score_z_entry=pos.score_z_entry,
                        bucket_entry=pos.bucket_entry,
                        funding_z_entry=pos.funding_z_entry,
                        oi_delta_entry=pos.oi_delta_entry,
                        sym_z_entry=pos.sym_z_entry,
                        partial_hit=pos.partial_done,
                        tp2_hit=tp2_hit_flag,
                        trail_used=trail_flag,
                    )
                    closed.append(trade)
                    trades_global.append(trade)
                    last_exit_time[symbol] = int(row.open_time)
                    if trade.net_ret <= 0:
                        last_loss_exit_time[symbol] = int(row.open_time)
                    # risk-off check
                    if recent_losses_exceeded():
                        risk_off_until_time = int(row.open_time + params.riskoff_cooldown_hours * 3_600_000) if params.riskoff_cooldown_hours else None
                    else:
                        # reset risk_off se lucro
                        if trade.net_ret > 0 and risk_off_until_time and row.open_time >= risk_off_until_time:
                            risk_off_until_time = None
                    positions.pop(symbol, None)
                    continue  # próximo loop

            # =========== Confirmação de sinal armado ===========
            if params.confirm_next_bar and symbol in armed_signals:
                arm = armed_signals[symbol]
                if row.open_time == arm.planned_entry_time:
                    # Validar filtros novamente
                    if risk_off_until_time and row.open_time < risk_off_until_time:
                        armed_signals.pop(symbol, None)
                    else:
                        # Filtro de perdas recentes por símbolo
                        if params.min_hours_since_loss_entry is not None and symbol in last_loss_exit_time:
                            delta_h = (row.open_time - last_loss_exit_time[symbol]) / 3_600_000
                            if delta_h < params.min_hours_since_loss_entry:
                                armed_signals.pop(symbol, None)
                                continue
                        # Filtros fatoriais na barra confirmadora
                        if not satisfies_factor_filters(row, params):
                            armed_signals.pop(symbol, None)
                            continue
                        # Entrada efetiva
                        hour_bucket = int(row.open_time // 3_600_000)
                        if params.max_new_entries_per_hour is not None:
                            if entries_per_hour.get(hour_bucket, 0) >= params.max_new_entries_per_hour:
                                armed_signals.pop(symbol, None)
                                continue
                        entry_price = row.close * (1 + params.slip_perc)
                        positions[symbol] = Position(
                            symbol=symbol,
                            entry_time=int(row.open_time),
                            entry_price=entry_price,
                            size=1.0,
                            peak_ret=0.0,
                            score_z_entry=arm.score_z_cs,
                            bucket_entry=arm.bucket,
                            funding_z_entry=arm.funding_z,
                            oi_delta_entry=arm.oi_delta,
                            sym_z_entry=arm.sym_z,
                            peak_price=entry_price
                        )
                        entries_per_hour[hour_bucket] = entries_per_hour.get(hour_bucket, 0) + 1
                        armed_signals.pop(symbol, None)

                elif row.open_time > arm.planned_entry_time:
                    # Expirou sem confirmar
                    armed_signals.pop(symbol, None)

            # =========== Preparar nova entrada (armar ou entrar) ===========
            if symbol not in positions:
                # Risk-off global
                if risk_off_until_time and row.open_time < risk_off_until_time:
                    continue
                # Cooldown simbólico
                last_exit = last_exit_time.get(symbol)
                if last_exit is not None:
                    hours_since = (row.open_time - last_exit) / 3_600_000
                    # dynamic cooldown (usamos último trade do símbolo)
                    cooldown_needed = params.cooldown_hours
                    if (params.cooldown_win_hours is not None or params.cooldown_sl_hours is not None):
                        for ct in reversed(closed):
                            if ct.symbol == symbol:
                                if ct.net_ret > 0:
                                    cooldown_needed = params.cooldown_win_hours if params.cooldown_win_hours is not None else cooldown_needed
                                else:
                                    cooldown_needed = params.cooldown_sl_hours if params.cooldown_sl_hours is not None else cooldown_needed
                                break
                    if hours_since < cooldown_needed:
                        continue
                # Filtro “loss since” por símbolo
                if params.min_hours_since_loss_entry is not None and symbol in last_loss_exit_time:
                    delta_h = (row.open_time - last_loss_exit_time[symbol]) / 3_600_000
                    if delta_h < params.min_hours_since_loss_entry:
                        continue
                # ATR filtro
                if params.min_atr_pct_entry is not None and (row.atr_pct < params.min_atr_pct_entry):
                    continue
                # Filtros básicos
                if not satisfies_basic(row, flt):
                    continue
                # Filtros fatoriais
                if not satisfies_factor_filters(row, params):
                    continue

                cond_entry = (
                    (params.mode == "bucket" and row.bucket == params.entry_bucket) or
                    (params.mode == "zscore" and row.score_z_cs >= params.z_entry)
                )
                if cond_entry:
                    if params.double_confirm:
                        ok_prev = True
                        prev_cs = row.get("score_z_cs_prev", np.nan)
                        if params.mode == "zscore" and not np.isnan(prev_cs):
                            ok_prev &= prev_cs >= params.z_entry
                        if params.require_symbol_z_confirm and "score_z_sym_prev" in row:
                            ok_prev &= row.score_z_sym_prev >= params.sym_z_entry_threshold
                        if not ok_prev:
                            continue
                    hour_bucket = int(row.open_time // 3_600_000)
                    if params.max_new_entries_per_hour is not None:
                        if entries_per_hour.get(hour_bucket, 0) >= params.max_new_entries_per_hour:
                            continue

                    if params.confirm_next_bar:
                        # Armar sinal
                        armed_signals[symbol] = ArmedSignal(
                            symbol=symbol,
                            arm_time=int(row.open_time),
                            planned_entry_time=int(nxt.open_time),
                            score_z_cs=row.score_z_cs,
                            bucket=int(row.bucket),
                            funding_z=row.get("funding_z", np.nan),
                            oi_delta=row.get("oi_delta_pct", np.nan),
                            sym_z=row.get("score_z_sym", np.nan),
                            entry_ref_close=row.close,
                            filters_snapshot={
                                "momentum": row.get(params.momentum_col),
                                "breakout": row.get(params.breakout_col),
                                "contrarian": row.get(params.contrarian_col)
                            },
                            score_pct_cs=row.get("score_pct_cs", np.nan)
                        )
                    else:
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
                            sym_z_entry=row.get("score_z_sym", np.nan),
                            peak_price=entry_price
                        )
                        entries_per_hour[hour_bucket] = entries_per_hour.get(hour_bucket, 0) + 1

        # Forced close final
        last = g.iloc[-1]
        if symbol in positions:
            pos = positions[symbol]
            gross_remaining = (last.close / pos.entry_price) - 1.0
            net_remaining = gross_remaining - (params.cost_perc + params.slip_perc)
            hold_h = int((last.open_time - pos.entry_time) // 3_600_000)
            total_net = pos.realized_partial_ret + net_remaining * pos.size
            total_gross = pos.realized_partial_ret + gross_remaining * pos.size
            trade = ClosedTrade(
                symbol=symbol,
                entry_time=pos.entry_time,
                exit_time=int(last.open_time),
                entry_price=pos.entry_price,
                exit_price=last.close,
                gross_ret=total_gross,
                net_ret=total_net,
                hold_hours=hold_h,
                reason="EOL",
                score_z_entry=pos.score_z_entry,
                bucket_entry=pos.bucket_entry,
                funding_z_entry=pos.funding_z_entry,
                oi_delta_entry=pos.oi_delta_entry,
                sym_z_entry=pos.sym_z_entry,
                partial_hit=pos.partial_done,
                tp2_hit=False,
                trail_used=False,
            )
            closed.append(trade)
            trades_global.append(trade)
            if trade.net_ret <= 0:
                last_loss_exit_time[symbol] = int(last.open_time)
            positions.pop(symbol, None)

    return closed


# ============================= SUMMARY ============================= #

def summarize(trades: list[ClosedTrade]):
    if not trades:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame([t.__dict__ for t in trades]).sort_values("exit_time")
    df["equity"] = (1 + df.net_ret).cumprod()
    roll_max = df.equity.cummax()
    dd = df.equity / roll_max - 1

    pos_profits = df.loc[df.net_ret > 0, "net_ret"].sort_values(ascending=False)
    half_sum = pos_profits.sum() * 0.5
    cum = pos_profits.cumsum()
    top_for_half = (cum <= half_sum).sum()

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
        "partial_trades": df.partial_hit.sum(),
        "tp2_trades": df.tp2_hit.sum(),
        "trail_trades": df.trail_used.sum(),
        "early_exits": (df.reason == "EARLY").sum(),
        "profit_conc_trades_for_50pct": top_for_half,
    }
    return pd.DataFrame([summary]), df


# ============================= ARGS ============================= #

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bucket", "zscore"], default="bucket")
    ap.add_argument("--buckets", type=int, default=5)
    ap.add_argument("--entry-bucket", type=int, default=None)
    ap.add_argument("--z-entry", type=float, default=0.8)
    ap.add_argument("--max-hold", type=int, default=24)

    # ATR-based
    ap.add_argument("--tp1-mult", type=float, default=1.0)
    ap.add_argument("--tp2-mult", type=float, default=2.0)
    ap.add_argument("--sl-mult", type=float, default=0.9)

    # Fixed targets / SL
    ap.add_argument("--fixed-tp1-pct", type=float, default=None)
    ap.add_argument("--fixed-tp2-pct", type=float, default=None)
    ap.add_argument("--fixed-sl-pct", type=float, default=None)

    # Partial fraction
    ap.add_argument("--partial-fraction", type=float, default=0.5,
                    help="Fração da posição encerrada no TP1 (0< f <1).")

    # Trailing
    ap.add_argument("--trailing-decay", type=float, default=0.5)
    ap.add_argument("--trail-requires-partial", action="store_true")
    ap.add_argument("--trail-after-fraction", type=float, default=None,
                    help="Fração entre TP1 e TP2 antes de permitir trailing.")
    ap.add_argument("--trail-dd-pct", type=float, default=None,
                    help="Drawdown absoluto (retorno) a partir do pico que aciona trailing.")
    ap.add_argument("--atr-window", type=int, default=14)

    # Early exit
    ap.add_argument("--early-exit", action="store_true")
    ap.add_argument("--early-z-threshold", type=float, default=-0.5)
    ap.add_argument("--early-min-hold", type=int, default=2)
    ap.add_argument("--use-symbol-z-for-early", action="store_true")

    # Double confirm
    ap.add_argument("--double-confirm", action="store_true")
    ap.add_argument("--require-symbol-z-confirm", action="store_true")
    ap.add_argument("--sym-z-entry-threshold", type=float, default=0.0)

    # Filtros fatoriais
    ap.add_argument("--min-momentum", type=float, default=None)
    ap.add_argument("--min-breakout", type=float, default=None)
    ap.add_argument("--min-contrarian", type=float, default=None)
    ap.add_argument("--momentum-col", type=str, default="momentum")
    ap.add_argument("--breakout-col", type=str, default="breakout")
    ap.add_argument("--contrarian-col", type=str, default="contrarian")

    # Filtros gerais
    ap.add_argument("--min-score-pct", type=float, default=0.9)
    ap.add_argument("--regimes", type=str, default="Vol_Expansao,Vol_Compressa,Tendencia_Alta")
    ap.add_argument("--regime-exclude", type=str, default="")
    ap.add_argument("--min-funding-z", type=float, default=-0.5)
    ap.add_argument("--min-oi-delta", type=float, default=None)
    ap.add_argument("--min-vol-pct", type=float, default=None)
    ap.add_argument("--req-score-nonneg", action="store_true")
    ap.add_argument("--min-atr-pct-entry", type=float, default=None)

    # Custos
    ap.add_argument("--cost-bps", type=float, default=10)
    ap.add_argument("--slip-bps", type=float, default=7)

    # Cooldown
    ap.add_argument("--cooldown-hours", type=float, default=0.0)
    ap.add_argument("--cooldown-win-hours", type=float, default=None)
    ap.add_argument("--cooldown-sl-hours", type=float, default=None)
    ap.add_argument("--min-hours-since-loss-entry", type=float, default=None)

    # Risk-off global
    ap.add_argument("--max-losses-window", type=int, default=None)
    ap.add_argument("--max-losses-allowed", type=int, default=None)
    ap.add_argument("--riskoff-cooldown-hours", type=float, default=None)

    # Entry pacing
    ap.add_argument("--max-new-entries-per-hour", type=int, default=None)

    # Confirm next bar
    ap.add_argument("--confirm-next-bar", action="store_true")

    ap.add_argument("--export-trades", action="store_true")
    return ap.parse_args()


# ============================= MAIN ============================= #

def main():
    args = parse_args()
    df = load_data(ENRICHED_PATH)
    flt = {
        "regimes": [r.strip() for r in args.regimes.split(",") if r.strip()],
        "exclude_regimes": [r.strip() for r in args.regime_exclude.split(",") if r.strip()],
        "min_score_pct": args.min_score_pct,
        "req_score_nonneg": args.req_score_nonneg,
        "min_funding_z": args.min_funding_z,
        "min_oi_delta": args.min_oi_delta,
        "min_vol_pct": args.min_vol_pct,
    }

    class P: ...
    p = P()
    # Copy args
    for k, v in vars(args).items():
        setattr(p, k.replace("-", "_"), v)
    # Normalização de nomes (alguns já usados antes)
    p.mode = args.mode
    p.entry_bucket = args.entry_bucket
    p.cost_perc = args.cost_bps / 10000.0
    p.slip_perc = args.slip_bps / 10000.0

    trades = backtest(df, p, flt)
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

    out_dir = Path("data/reports"); out_dir.mkdir(exist_ok=True, parents=True)
    summary.to_parquet(out_dir / "backtest_summary_adv.parquet", index=False)
    if args.export_trades:
        trades_df.to_parquet(out_dir / "backtest_trades_adv.parquet", index=False)
        print(f"\nTrades salvas em: {out_dir / 'backtest_trades_adv.parquet'}")


if __name__ == "__main__":
    main()
