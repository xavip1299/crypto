import asyncio
import yaml
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# Imports internos
from core.datasource import fetch_multiple
from core.features import build_feature_set
from core.regimes import RegimeClassifier
from core.scoring import ScoringEngine
from core.storage import ParquetStore
from core.utils import setup_logging
from core.exchanges import load_derivatives  # funding & OI (perps)

# Placeholder de alerta (futuro: Telegram)
def send_alert(message: str):
    logging.getLogger("alert").info(message)


async def enrich_with_derivatives(df: pd.DataFrame, funding_item, oi_hist):
    """
    Adiciona funding (último) e open interest histórico aproximado ao DataFrame OHLCV.
    Sinaliza apenas a última linha para funding; OI será alinhado por timestamp.
    """
    if funding_item:
        # Adiciona funding somente na última linha
        df.loc[df.index[-1], "funding"] = funding_item.get("last_funding_rate")

    if oi_hist:
        oi_df = pd.DataFrame(oi_hist)
        if not oi_df.empty:
            # Tentar merge aproximado por timestamp (open_time é epoch ms)
            # Converter open_time para int ms se ainda não estiver
            # open_time (Binance klines) já vem em ms
            oi_df = oi_df.sort_values("timestamp")
            df = df.copy()
            df["open_interest_usd"] = None
            # Simple alignment: nearest timestamp
            # (poderia melhorar usando merge_asof)
            for i, row in df.iterrows():
                ts = row["open_time"]
                # Encontrar entrada de OI com timestamp <= ts (último conhecido)
                subset = oi_df[oi_df["timestamp"] <= ts]
                if not subset.empty:
                    df.at[i, "open_interest_usd"] = subset.iloc[-1]["open_interest_usd"]
            df["open_interest_usd"] = pd.to_numeric(df["open_interest_usd"])
    return df


async def process_symbol(
    symbol: str,
    interval: str,
    limit: int,
    storage_path: str,
    funding_map: dict,
    oi_map: dict,
) -> pd.DataFrame:
    # Buscar OHLCV
    raw_map = await fetch_multiple([symbol], interval, limit)
    rows = raw_map.get(symbol, [])
    df = pd.DataFrame(rows)
    store = ParquetStore(storage_path, symbol, interval)
    df = store.append_and_dedupe(df)

    # Dados derivativos (se símbolo em perps)
    funding_item = next(
        (f for f in funding_map if f["symbol"] == symbol), None
    )
    oi_hist = oi_map.get(symbol)

    df = await enrich_with_derivatives(df, funding_item, oi_hist)

    # Features
    df = build_feature_set(df)
    return df


async def main():
    settings = yaml.safe_load(open("config/settings.yaml"))
    setup_logging(settings.get("logging", {}).get("level", "INFO"))

    spot_symbols = settings["universe"]["spot"]
    perp_symbols = settings["universe"].get("perps", [])
    interval = settings["data"]["timeframe"]
    limit = settings["data"]["ohlcv_limit"]
    storage_path = settings["storage"]["path"]

    # Carregar funding & OI apenas para símbolos perp
    funding_list = []
    oi_map = {}
    if perp_symbols:
        funding_list, oi_map = await load_derivatives(perp_symbols, period="1h", limit=72)

    # Processar símbolos (spot e perps — perps podem estar em ambas listas)
    tasks = [
        process_symbol(
            sym, interval, limit, storage_path, funding_list, oi_map
        )
        for sym in spot_symbols
    ]
    results = await asyncio.gather(*tasks)

    clf = RegimeClassifier(
        vol_short_window=settings["regime"]["vol_short_window"],
        vol_long_window=settings["regime"]["vol_long_window"],
        compress_ratio=settings["regime"]["compress_ratio"],
        expansion_ratio=settings["regime"]["expansion_ratio"],
    )
    scorer = ScoringEngine(settings["scoring"])

    snapshots = []
    for sym, df in zip(spot_symbols, results):
        if df.empty:
            continue
        regime = clf.classify(df)
        last_row = df.iloc[-1].to_dict()
        sb = scorer.compute(last_row, regime)
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

    snap_df = pd.DataFrame(snapshots).sort_values("score", ascending=False)
    print("\n=== SCORE SNAPSHOT @", datetime.now(timezone.utc).isoformat(), "===")
    print(snap_df.to_string(index=False))

    # Alerta de top 1 (placeholder)
    if not snap_df.empty:
        top = snap_df.iloc[0]
        send_alert(
            f"TOP SIGNAL {top.symbol} score={top.score:.4f} regime={top.regime} "
            f"momentum={top.momentum:.4f} breakout={top.breakout:.4f} contrarian={top.contrarian:.4f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
