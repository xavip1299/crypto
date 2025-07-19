import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

# Core (internos)
from core.datasource import fetch_multiple
from core.features import build_feature_set, recompute_derivative_features
from core.regimes import RegimeClassifier
from core.scoring import ScoringEngine
from core.storage import ParquetStore
from core.exchanges import load_derivatives
from core.alerts import TelegramAlerter
from core.historical import paginate_klines, interval_to_ms


# ===================== Config Helpers ===================== #

def load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings() -> dict:
    return load_yaml("config/settings.yaml")


def load_secrets() -> dict:
    try:
        return load_yaml("config/secrets.yaml")
    except FileNotFoundError:
        return {}


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# ===================== Derivatives Enrichment (live) ===================== #

async def enrich_with_derivatives(df: pd.DataFrame, funding_item, oi_hist):
    """
    Adiciona funding (apenas última linha) e open_interest_usd via alinhamento simplificado.
    (Para histórico completo usamos merge_asof mais robusto em integrate.py)
    """
    if df.empty:
        return df

    if funding_item:
        df.loc[df.index[-1], "funding"] = funding_item.get("last_funding_rate")

    if oi_hist:
        oi_df = pd.DataFrame(oi_hist).sort_values("timestamp")
        if not oi_df.empty and "open_time" in df.columns:
            df = df.copy()
            df["open_interest_usd"] = None
            for i, row in df.iterrows():
                ts = row["open_time"]
                subset = oi_df[oi_df["timestamp"] <= ts]
                if not subset.empty:
                    df.at[i, "open_interest_usd"] = subset.iloc[-1]["open_interest_usd"]
            df["open_interest_usd"] = pd.to_numeric(df["open_interest_usd"])
    return df


# ===================== Symbol Processing (live incremental) ===================== #

async def process_symbol(symbol: str,
                         interval: str,
                         limit: int,
                         storage_path: str,
                         funding_list,
                         oi_map) -> pd.DataFrame:
    """
    Busca OHLCV recente, persiste incrementalmente e enriquece features + derivativos (live).
    """
    raw_map = await fetch_multiple([symbol], interval, limit)
    rows = raw_map.get(symbol, [])
    df = pd.DataFrame(rows)

    store = ParquetStore(storage_path, symbol, interval)
    df = store.append_and_dedupe(df)

    funding_item = next((f for f in funding_list if f["symbol"] == symbol), None)
    oi_hist = oi_map.get(symbol)
    df = await enrich_with_derivatives(df, funding_item, oi_hist)
    df = build_feature_set(df)
    return df


# ===================== Bootstrap Historical ===================== #

async def bootstrap_history(symbols, interval, target_hours, step_limit, storage_path):
    """
    Faz backfill histórico (retrocessão) por símbolo usando paginação.
    """
    logging.info("Bootstrap histórico: target_hours=%s interval=%s step=%s",
                 target_hours, interval, step_limit)
    tf_ms = interval_to_ms(interval)
    candles_needed = int((target_hours * 3600_000) / tf_ms)
    logging.info("Candles estimados por símbolo: %s", candles_needed)

    for sym in symbols:
        logging.info("Bootstrap %s...", sym)
        rows = await paginate_klines(sym, interval, candles_needed, step=step_limit)
        if not rows:
            logging.warning("Sem dados bootstrap para %s", sym)
            continue
        df_boot = pd.DataFrame(rows)
        store = ParquetStore(storage_path, sym, interval)
        store.append_and_dedupe(df_boot)
        logging.info("Bootstrap %s concluído (%d candles).", sym, len(df_boot))


# ===================== Histórico Derivativos (Funding & OI) ===================== #

async def build_full_derivatives(perp_symbols, deriv_cfg, force_rebuild: bool):
    """
    Orquestra coleta/persistência de funding & OI históricos via derivatives_history.
    Retorna dict com dataframes por símbolo:
      {
        "funding": {SYM: df},
        "oi": {SYM: df}
      }
    """
    if not perp_symbols:
        return None
    from core.derivatives_history import build_derivatives_history
    logging.info("Construindo histórico de derivados (force=%s)...", force_rebuild)
    return await build_derivatives_history(perp_symbols, deriv_cfg, force_rebuild=force_rebuild)


def recompute_full_price_with_derivatives(spot_symbols,
                                          storage_path,
                                          interval,
                                          deriv_hist,
                                          regime_cfg,
                                          scoring_cfg,
                                          deriv_cfg):
    """
    Re-merge funding & OI históricos em cada série de preços, recalcula features derivativas
    e (se solicitado) gera mapa de DataFrames para posterior backtest.
    """
    if not deriv_hist:
        return {}

    from core.integrate import merge_derivatives_into_price

    extended_price_map = {}
    for sym in spot_symbols:
        store = ParquetStore(storage_path, sym, interval)
        base_df = store.load()
        if base_df.empty:
            continue
        funding_df = deriv_hist["funding"].get(sym)
        oi_df = deriv_hist["oi"].get(sym)
        merged = merge_derivatives_into_price(base_df, funding_df, oi_df)
        merged = build_feature_set(merged)            # inclui ret, etc.
        merged = recompute_derivative_features(merged)  # funding_z, oi_delta_pct
        # Persist back
        try:
            merged.to_parquet(store.parquet_path, index=False)
        except Exception:
            merged.to_csv(store.csv_path, index=False)
        extended_price_map[sym] = merged
    return extended_price_map


# ===================== Historical Signals Backtest ===================== #

def build_historical_signals(extended_price_map, regime_cfg, scoring_cfg, output_path: Path):
    from core.backtest_signals import generate_historical_signals
    clf_bt = RegimeClassifier(
        vol_short_window=regime_cfg["vol_short_window"],
        vol_long_window=regime_cfg["vol_long_window"],
        compress_ratio=regime_cfg["compress_ratio"],
        expansion_ratio=regime_cfg["expansion_ratio"],
    )
    scorer_bt = ScoringEngine(scoring_cfg)
    sig_df = generate_historical_signals(extended_price_map, clf_bt, scorer_bt)
    sig_df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    logging.info("Sinais históricos gerados (%d linhas) -> %s", len(sig_df), output_path)


# ===================== Main Flow ===================== #

async def main():
    settings = load_settings()
    secrets = load_secrets()

    setup_logging(settings.get("logging", {}).get("level", "INFO"))

    spot_symbols = settings["universe"]["spot"]
    perp_symbols = settings["universe"].get("perps", [])
    interval = settings["data"]["timeframe"]
    limit = settings["data"]["ohlcv_limit"]
    storage_path = settings["storage"]["path"]

    regime_cfg = settings["regime"]
    scoring_cfg = settings["scoring"]

    # CLI Flags
    bootstrap_cfg = settings.get("bootstrap", {})
    bootstrap_flag = "--bootstrap" in sys.argv or bootstrap_cfg.get("enabled", False)
    force_deriv = "--rebuild-derivatives" in sys.argv
    hist_signals_flag = "--build-hist-signals" in sys.argv

    deriv_cfg = settings.get("derivatives_history", {})
    deriv_enabled = deriv_cfg.get("enabled", False)

    # (1) Bootstrap histórico de preços (se solicitado)
    if bootstrap_flag:
        await bootstrap_history(
            spot_symbols,
            interval,
            bootstrap_cfg.get("target_hours", 2000),
            bootstrap_cfg.get("step_limit", 1000),
            storage_path,
        )

    # (2) Histórico de derivados (funding & OI)
    deriv_hist = None
    if deriv_enabled:
        deriv_hist = await build_full_derivatives(perp_symbols, deriv_cfg, force_rebuild=force_deriv)

    # (3) Re-merge & recompute features históricas (se derivativos habilitados)
    extended_price_map = {}
    if deriv_hist and deriv_cfg.get("recompute_features", True):
        extended_price_map = recompute_full_price_with_derivatives(
            spot_symbols,
            storage_path,
            interval,
            deriv_hist,
            regime_cfg,
            scoring_cfg,
            deriv_cfg
        )

    # (4) Geração opcional de sinais históricos (backtest offline)
    if hist_signals_flag and extended_price_map:
        reports_cfg = settings.get("reports", {})
        reports_path = Path(reports_cfg.get("path", "data/reports"))
        reports_path.mkdir(parents=True, exist_ok=True)
        hist_signals_path = reports_path / "historical_signals.parquet"
        build_historical_signals(extended_price_map, regime_cfg, scoring_cfg, hist_signals_path)

    # (5) Coleta “live” incremental + snapshot
    funding_list = []
    oi_map = {}
    if perp_symbols:
        # Uso realtime (último funding & pequeno histórico OI)
        funding_list, oi_map = await load_derivatives(perp_symbols, period="1h", limit=72)

    # Kill switch simples se precisava derivados e não retornou nada
    if perp_symbols and deriv_enabled and not funding_list and not force_deriv:
        logging.warning("Kill switch: nenhum funding realtime retornado. Abortando ciclo live.")
        return

    tasks = [
        process_symbol(sym, interval, limit, storage_path, funding_list, oi_map)
        for sym in spot_symbols
    ]
    results = await asyncio.gather(*tasks)

    clf_live = RegimeClassifier(
        vol_short_window=regime_cfg["vol_short_window"],
        vol_long_window=regime_cfg["vol_long_window"],
        compress_ratio=regime_cfg["compress_ratio"],
        expansion_ratio=regime_cfg["expansion_ratio"],
    )
    scorer_live = ScoringEngine(scoring_cfg)

    snapshots = []
    for sym, df in zip(spot_symbols, results):
        if df.empty:
            continue
        regime = clf_live.classify(df)
        last_row = df.iloc[-1].to_dict()
        sb = scorer_live.compute(last_row, regime)
        snapshots.append(
            {
                "symbol": sym,
                "regime": regime,
                "score": sb.total,
                "momentum": sb.momentum,
                "breakout": sb.breakout,
                "contrarian": sb.contrarian,
                "penalty": sb.penalty,
                "funding_z": last_row.get("funding_z"),
                "oi_delta_pct": last_row.get("oi_delta_pct"),
            }
        )

    if not snapshots:
        logging.warning("Nenhum snapshot gerado.")
        return

    snap_df = pd.DataFrame(snapshots).sort_values("score", ascending=False)

    print("\n=== SCORE SNAPSHOT @", datetime.now(timezone.utc).isoformat(), "===")
    print(
        snap_df[
            [
                "symbol",
                "regime",
                "score",
                "momentum",
                "breakout",
                "contrarian",
                "penalty",
                "funding_z",
                "oi_delta_pct",
            ]
        ].to_string(index=False)
    )

    # (6) Persistência snapshot acumulado
    reports_cfg = settings.get("reports", {})
    reports_path = Path(reports_cfg.get("path", "data/reports"))
    snapshot_file = reports_cfg.get("snapshot_file", "snapshots.parquet")
    reports_path.mkdir(parents=True, exist_ok=True)
    report_path = reports_path / snapshot_file

    current_snapshot = snap_df.copy()
    current_snapshot["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    if report_path.exists():
        try:
            old = pd.read_parquet(report_path)
            combined = pd.concat([old, current_snapshot], ignore_index=True)
            combined = combined.drop_duplicates(subset=["symbol", "timestamp_utc"])
        except Exception:
            combined = current_snapshot
    else:
        combined = current_snapshot

    try:
        combined.to_parquet(report_path, engine="pyarrow", compression="snappy", index=False)
        logging.info("Snapshot salvo (%d total linhas) -> %s", len(combined), report_path)
    except Exception as e:
        logging.warning("Falha ao salvar snapshot: %s", e)

    # (7) Alertas Telegram
    tg_cfg = secrets.get("telegram") if secrets else None
    if tg_cfg and tg_cfg.get("bot_token") and tg_cfg.get("chat_id"):
        alerter = TelegramAlerter(
            bot_token=str(tg_cfg["bot_token"]).strip(),
            chat_id=tg_cfg["chat_id"],
            parse_mode="Markdown",
        )
        top_n = snap_df.head(3)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [f"*Top Signals — cry2muchbot* ({ts})"]
        for _, row in top_n.iterrows():
            lines.append(
                f"`{row.symbol}` score={row.score:.5f} mom={row.momentum:.5f} "
                f"brk={row.breakout:.3f} ctr={row.contrarian:.3f}"
            )
        msg = "\n".join(lines)
        success = await alerter.send(msg)
        if success:
            logging.info("Alerta Telegram enviado.")
        else:
            logging.warning("Falha ao enviar alerta Telegram.")
    else:
        logging.info("Telegram não configurado (pulei envio).")


if __name__ == "__main__":
    asyncio.run(main())
