"""Ingestão de dados de múltiplas fontes (multi-venue VWAP, redundância)."""
import aiohttp
from typing import Tuple

class DataSourceManager:
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    async def fetch_binance_klines(self, symbol: str, interval: str = "1h", limit: int = 500) -> List[Dict[str, Any]]:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        async with self.session.get(url, timeout=10) as r:
            data = await r.json()
            return [
                {
                    "open_time": x[0],
                    "open": float(x[1]),
                    "high": float(x[2]),
                    "low": float(x[3]),
                    "close": float(x[4]),
                    "volume": float(x[5]),
                    "close_time": x[6]
                } for x in data
            ]

    async def multi_venue_vwap(self, symbol: str) -> Optional[float]:
        # Placeholder: combinar preços spot de venues habilitadas.
        # Futuro: adicionar OKX, coinbase, etc.; fallback se uma falhar.
        try:
            kl = await self.fetch_binance_klines(symbol, interval="1m", limit=1)
            if not kl:
                return None
            return kl[-1]["close"]  # simplificado
        except Exception as e:
            LOG.warning(f"VWAP fetch failed {symbol}: {e}")
            return None

# ========================= core/features.py =========================
"""Cálculo de features (retornos, volatilidade, funding z, OI delta, breadth, microestrutura)."""
import pandas as pd
import numpy as np
from typing import Mapping

class FeatureEngineer:
    def __init__(self):
        pass

    def basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ret_1'] = df['close'].pct_change()
        df['ret_24'] = df['close'].pct_change(24)
        df['ret_4'] = df['close'].pct_change(4)
        df['vol_real_24'] = df['ret_1'].rolling(24).std()*np.sqrt(24)
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        df['range_expansion'] = (df['high'] - df['low'])/df['atr']
        return df

    def add_zscores(self, df: pd.DataFrame, cols: List[str], win: int = 30) -> pd.DataFrame:
        for c in cols:
            rolling = df[c].rolling(win)
            df[c+"_z"] = (df[c] - rolling.mean())/(rolling.std()+1e-9)
        return df

# ========================= core/regimes.py =========================
from enum import Enum
import pandas as pd

class Regime(str, Enum):
    VOL_COMPRESSA = "Vol_Compressa"
    VOL_EXPANSAO = "Vol_Expansao"
    TEND_ALTA = "Tendencia_Alta"
    TEND_BAIXA = "Tendencia_Baixa"
    CHOP = "Chop"

class RegimeClassifier:
    def classify(self, df: pd.DataFrame) -> Regime:
        # Heurística simples (placeholder). Expandir com clustering probabilístico.
        vol7 = df['ret_1'].rolling(7).std().iloc[-1]
        vol30 = df['ret_1'].rolling(30).std().iloc[-1]
        close = df['close'].iloc[-1]
        ma50 = df['close'].rolling(50).mean().iloc[-1]
        slope = ma50 - df['close'].rolling(50).mean().shift(5).iloc[-1]
        if vol7 < vol30*0.6: return Regime.VOL_COMPRESSA
        if vol7 > vol30*1.3: return Regime.VOL_EXPANSAO
        if close > ma50 and slope > 0: return Regime.TEND_ALTA
        if close < ma50 and slope < 0: return Regime.TEND_BAIXA
        return Regime.CHOP
