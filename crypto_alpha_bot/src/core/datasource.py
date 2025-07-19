import aiohttp
import asyncio
import logging
from typing import List, Dict, Any, Optional

LOG = logging.getLogger("datasource")

BINANCE_BASE = "https://api.binance.com"


async def fetch_klines_window(session, symbol: str, interval: str,
                              limit: int = 1000,
                              start_time: Optional[int] = None,
                              end_time: Optional[int] = None):
    """
    Busca klines com janela opcional (start_time inclusive, end_time exclusivo).
    Timestamps em milissegundos.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time
    url = f"{BINANCE_BASE}/api/v3/klines"
    try:
        async with session.get(url, params=params, timeout=15) as r:
            r.raise_for_status()
            data = await r.json()
            rows = []
            for k in data:
                rows.append(
                    {
                        "open_time": k[0],
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "close_time": k[6],
                    }
                )
            return rows
    except Exception as e:
        LOG.warning("Erro klines window %s: %s", symbol, e)
        return []


async def fetch_klines(session, symbol: str, interval: str, limit: int = 500):
    return await fetch_klines_window(session, symbol, interval, limit)


async def fetch_multiple(symbols: List[str], interval: str, limit: int) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_klines(session, s, interval, limit) for s in symbols]
        results = await asyncio.gather(*tasks)
    return {sym: rows for sym, rows in zip(symbols, results)}
