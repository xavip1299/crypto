from typing import Dict, Any, NamedTuple
from core.regimes import Regime

class ScoreBreakdown(NamedTuple):
    total: float
    momentum: float
    breakout: float
    contrarian: float
    penalty: float
    extras: Dict[str, float]

class ScoringEngine:
    def __init__(self, weights: Dict[str, float]):
        self.w = weights

    def compute(self, row: Dict[str, Any], regime: Regime) -> ScoreBreakdown:
        momentum = (row.get("ret_4", 0) + 0.5 * row.get("ret_24", 0)) * row.get("mom_quality", 1)
        breakout = 0.0
        if row.get("above_pivot_high") and row.get("range_expansion", 0) > 1.2:
            breakout = row.get("range_expansion", 0)
        contrarian = 0.0
        if "funding_z" in row and abs(row.get("ret_4", 0)) < 0.002 and row["funding_z"] > 2:
            contrarian += 0.02 * row["funding_z"]
        if "oi_delta_pct" in row and row.get("ret_24", 0) > 0 and row["oi_delta_pct"] < -0.05:
            contrarian += 0.01
        overext = max(0, row.get("overext_z", 0) - 2)
        base = (
            self.w["momentum_w"] * momentum
            + self.w["breakout_w"] * breakout
            + self.w["contrarian_w"] * contrarian
            - 0.05 * overext
        )
        if regime == Regime.VOL_COMPRESSA:
            base += (1 - min(1, row.get("rci_20", 1))) * 0.05
        elif regime == Regime.TEND_BAIXA:
            base -= max(0, momentum)
        extras = {
            "rci_20": row.get("rci_20", 0),
            "overext_z": row.get("overext_z", 0),
        }
        return ScoreBreakdown(
            total=base,
            momentum=momentum,
            breakout=breakout,
            contrarian=contrarian,
            penalty=overext,
            extras=extras,
        )
