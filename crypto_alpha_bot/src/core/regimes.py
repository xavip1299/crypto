from enum import Enum
import pandas as pd

class Regime(str, Enum):
    VOL_COMPRESSA = "Vol_Compressa"
    VOL_EXPANSAO = "Vol_Expansao"
    TEND_ALTA = "Tendencia_Alta"
    TEND_BAIXA = "Tendencia_Baixa"
    CHOP = "Chop"

class RegimeClassifier:
    def __init__(
        self,
        vol_short_window: int = 24,
        vol_long_window: int = 120,
        compress_ratio: float = 0.6,
        expansion_ratio: float = 1.3,
    ):
        self.ws = vol_short_window
        self.wl = vol_long_window
        self.comp = compress_ratio
        self.exp = expansion_ratio

    def classify(self, df: pd.DataFrame) -> Regime:
        if len(df) < self.wl + 5:
            return Regime.CHOP
        ret = df["close"].pct_change()
        vol_s = ret.rolling(self.ws).std().iloc[-1]
        vol_l = ret.rolling(self.wl).std().iloc[-1]
        close = df["close"].iloc[-1]
        ma = df["close"].rolling(50).mean().iloc[-1]
        slope = df["close"].rolling(50).mean().iloc[-1] - df["close"].rolling(50).mean().shift(5).iloc[-1]
        ratio = (vol_s / (vol_l + 1e-9)) if vol_l else 1
        if ratio < self.comp:
            return Regime.VOL_COMPRESSA
        if ratio > self.exp:
            return Regime.VOL_EXPANSAO
        if close > ma and slope > 0:
            return Regime.TEND_ALTA
        if close < ma and slope < 0:
            return Regime.TEND_BAIXA
        return Regime.CHOP
