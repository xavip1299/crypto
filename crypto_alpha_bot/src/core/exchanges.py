import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional, Tuple

LOG = logging.getLogger("exchanges")
BINANCE_PERP_BASE = "https://fapi.binance.com"


class BinancePerpClient:
    """
    Cliente para endpoints principais de perpetual futures (funding rate + open interest).
    """

    def __init__(self, session: aiohttp.ClientSession, symbols: List[str]):
        self.session = session
        self.symbols = symbols

    async def fetch_realtime_funding(self, symbol: str) -> Optional[Dict[str, Any]]:
        url = f"{BINANCE_PERP_BASE}/fapi/v1/premiumIndex?symbol={symbol}"
        try:
            async with self.session.get(url, timeout=10) as r:
                r.raise_for_status()
                data = await r.json()
                return {
                    "symbol": symbol,
                    "mark_price": float(data["markPrice"]),
                    "index_price": float(data["indexPrice"]),
                    "last_funding_rate": float(data["lastFundingRate"]),
                    "next_funding_time": int(data["nextFundingTime"]),
                }
        except Exception as e:
            LOG.warning("Funding erro %s: %s", symbol, e)
            return None

    async def fetch_open_interest(
        self, symbol: str, period: str = "1h", limit: int = 72
    ) -> Optional[List[Dict[str, Any]]]:
        url = (
            f"{BINANCE_PERP_BASE}/futures/data/openInterestHist?"
            f"symbol={symbol}&period={period}&limit={limit}"
        )
        try:
            async with self.session.get(url, timeout=10) as r:
                r.raise_for_status()
                data = await r.json()
                return [
                    {
                        "symbol": symbol,
                        "timestamp": int(d["timestamp"]),
                        "open_interest_contracts": float(d["sumOpenInterest"]),
                        "open_interest_usd": float(d["sumOpenInterestValue"]),
                    }
                    for d in data
                ]
        except Exception as e:
            LOG.warning("OI erro %s: %s", symbol, e)
            return None

    async def batch_realtime_funding(self) -> List[Dict[str, Any]]:
        results = await asyncio.gather(
            *[self.fetch_realtime_funding(s) for s in self.symbols]
        )
        return [r for r in results if r]

    async def fetch_all_oi(self, period: str = "1h", limit: int = 72) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for sym in self.symbols:
            hist = await self.fetch_open_interest(sym, period=period, limit=limit)
            if hist:
                out[sym] = hist
        return out


async def load_derivatives(
    symbols: List[str], period: str = "1h", limit: int = 72
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    async with aiohttp.ClientSession() as session:
        client = BinancePerpClient(session, symbols)
        funding = await client.batch_realtime_funding()
        oi_map = await client.fetch_all_oi(period=period, limit=limit)
    return funding, oi_map
