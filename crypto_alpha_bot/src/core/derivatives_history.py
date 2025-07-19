import aiohttp
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

LOG = logging.getLogger("derivatives_history")
BINANCE_PERP_BASE = "https://fapi.binance.com"

# ---------- Helpers de tempo ----------
def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

# ---------- Funding History ----------
async def fetch_funding_window(session: aiohttp.ClientSession,
                               symbol: str,
                               start_ms: int,
                               end_ms: int,
                               limit: int = 1000) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit
    }
    url = f"{BINANCE_PERP_BASE}/fapi/v1/fundingRate"
    try:
        async with session.get(url, params=params, timeout=15) as r:
            r.raise_for_status()
            data = await r.json()
            if not data:
                return pd.DataFrame()
            rows = []
            for d in data:
                rows.append({
                    "symbol": d["symbol"],
                    "funding_time": int(d["fundingTime"]),
                    "funding_rate": float(d["fundingRate"]),
                })
            return pd.DataFrame(rows)
    except Exception as e:
        LOG.warning("Funding window erro %s: %s", symbol, e)
        return pd.DataFrame()

async def fetch_full_funding(symbol: str,
                             lookback_days: int,
                             step_hours: int) -> pd.DataFrame:
    """
    Varre em janelas retroativas.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    total_hours = int((end - start).total_seconds() // 3600)
    steps = []
    current_end = end
    while current_end > start:
        current_start = current_end - timedelta(hours=step_hours)
        if current_start < start:
            current_start = start
        steps.append((current_start, current_end))
        current_end = current_start
    steps.reverse()

    out = []
    async with aiohttp.ClientSession() as session:
        for (s, e) in steps:
            df = await fetch_funding_window(session, symbol, dt_to_ms(s), dt_to_ms(e))
            if not df.empty:
                out.append(df)
            await asyncio.sleep(0.2)
    if not out:
        return pd.DataFrame()
    full = pd.concat(out, ignore_index=True)
    full = full.drop_duplicates(subset=["funding_time"]).sort_values("funding_time")
    return full

# ---------- Open Interest History ----------
async def fetch_oi_window(session: aiohttp.ClientSession,
                          symbol: str,
                          period: str,
                          start_ms: int,
                          end_ms: int) -> pd.DataFrame:
    """
    Endpoint não documenta 'startTime'/'endTime' nos params oficiais,
    porém testamos se aceita. Caso não responda adequadamente,
    fallback: apenas uso sem start/end (limitação).
    """
    # Tentativa de parâmetros
    params = {
        "symbol": symbol,
        "period": period,
        "limit": 500
    }
    # Observação: se start/end não funcionarem, remover abaixo.
    params["startTime"] = start_ms
    params["endTime"] = end_ms

    url = f"{BINANCE_PERP_BASE}/futures/data/openInterestHist"
    try:
        async with session.get(url, params=params, timeout=15) as r:
            r.raise_for_status()
            data = await r.json()
            if not data:
                return pd.DataFrame()
            rows = []
            for d in data:
                rows.append({
                    "symbol": d["symbol"],
                    "timestamp": int(d["timestamp"]),
                    "open_interest_contracts": float(d["sumOpenInterest"]),
                    "open_interest_usd": float(d["sumOpenInterestValue"]),
                })
            return pd.DataFrame(rows)
    except Exception as e:
        LOG.warning("OI window erro %s: %s", symbol, e)
        return pd.DataFrame()

async def fetch_full_oi(symbol: str,
                        lookback_days: int,
                        step_hours: int,
                        period: str = "1h") -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    steps = []
    current_end = end
    while current_end > start:
        current_start = current_end - timedelta(hours=step_hours)
        if current_start < start:
            current_start = start
        steps.append((current_start, current_end))
        current_end = current_start
    steps.reverse()

    out = []
    async with aiohttp.ClientSession() as session:
        for (s, e) in steps:
            df = await fetch_oi_window(session, symbol, period, dt_to_ms(s), dt_to_ms(e))
            if not df.empty:
                out.append(df)
            await asyncio.sleep(0.25)
    if not out:
        return pd.DataFrame()
    full = pd.concat(out, ignore_index=True)
    full = full.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return full

# ---------- Persistência / Orquestração ----------
def load_cached(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

def merge_incremental(old: pd.DataFrame, new: pd.DataFrame, key: str) -> pd.DataFrame:
    if old.empty:
        return new
    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=[key]).sort_values(key)
    return combined

async def build_derivatives_history(symbols: List[str],
                                    cfg: dict,
                                    force_rebuild: bool = False) -> dict:
    paths_cfg = cfg["paths"]
    base = Path(paths_cfg["base"])
    funding_dir = base / paths_cfg["funding_dir"]
    oi_dir = base / paths_cfg["oi_dir"]
    funding_dir.mkdir(parents=True, exist_ok=True)
    oi_dir.mkdir(parents=True, exist_ok=True)

    out = {"funding": {}, "oi": {}}

    for sym in symbols:
        # Funding
        fund_path = funding_dir / f"{sym}_funding.parquet"
        old_funding = load_cached(fund_path)
        if force_rebuild or old_funding.empty:
            LOG.info("Baixando funding completo %s...", sym)
            new_f = await fetch_full_funding(sym,
                                             cfg["funding"]["lookback_days"],
                                             cfg["funding"]["step_hours"])
            if new_f.empty:
                LOG.warning("Funding vazio %s", sym)
            else:
                new_f.to_parquet(fund_path, index=False)
                out["funding"][sym] = new_f
        else:
            LOG.info("Usando funding cache %s (%d linhas)", sym, len(old_funding))
            out["funding"][sym] = old_funding

        # OI
        oi_path = oi_dir / f"{sym}_oi.parquet"
        old_oi = load_cached(oi_path)
        if force_rebuild or old_oi.empty:
            LOG.info("Baixando OI completo %s...", sym)
            new_oi = await fetch_full_oi(sym,
                                         cfg["oi"]["lookback_days"],
                                         cfg["oi"]["step_hours"],
                                         period="1h")
            if new_oi.empty:
                LOG.warning("OI vazio %s", sym)
            else:
                new_oi.to_parquet(oi_path, index=False)
                out["oi"][sym] = new_oi
        else:
            LOG.info("Usando OI cache %s (%d linhas)", sym, len(old_oi))
            out["oi"][sym] = old_oi

    return out
