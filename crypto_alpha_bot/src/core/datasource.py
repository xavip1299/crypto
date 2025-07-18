import aiohttp, asyncio, logging
from typing import List, Dict, Any, Optional

LOG = logging.getLogger("datasource")
BASE_URL = "https://api.binance.com"

class DataSource:
    """Ingestão OHLCV spot (Binance). Pode expandir depois para multi-venue."""
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    async def fetch_klines(self, symbol: str, interval: str = "1h", limit: int = 500) -> List[Dict[str, Any]]:
        url = f"{BASE_URL}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        async with self.session.get(url, timeout=15) as r:
            r.raise_for_status()
            data = await r.json()
            return [
                {
                    "symbol": symbol,
                    "open_time": row[0],
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "close_time": row[6],
                }
                for row in data
            ]

async def fetch_multiple(symbols: List[str], interval: str, limit: int) -> Dict[str, List[Dict[str, Any]]]:
    async with DataSource() as ds:
        tasks = [ds.fetch_klines(sym, interval, limit) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    out = {}
    for sym, res in zip(symbols, results):
        if isinstance(res, Exception):
            LOG.warning("Erro ao buscar %s: %s", sym, res)
        else:
            out[sym] = res
    return out
