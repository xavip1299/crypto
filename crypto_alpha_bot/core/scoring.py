from typing import NamedTuple

class ScoreBreakdown(NamedTuple):
    total: float
    momentum: float
    breakout: float
    contrarian: float
    penalty: float

class ScoringEngine:
    def __init__(self, weights: Dict[str,float]):
        self.w = weights

    def compute(self, row: Dict[str, Any], regime: Regime) -> ScoreBreakdown:
        momentum = row.get('ret_4',0) + 0.5*row.get('ret_24',0)
        breakout = (row.get('range_expansion',0) if row.get('range_expansion',0) > 1.2 else 0)
        contrarian = 0  # placeholder (usar funding_z, divergências)
        overext = max(0, row.get('ret_24',0) - 0.05)
        base = (self.w['momentum_w']*momentum + self.w['breakout_w']*breakout + self.w['contrarian_w']*contrarian) - 0.5*overext
        # Ajuste por regime (exemplo)
        if regime == Regime.VOL_COMPRESSA:
            base += 0.1 * (row.get('range_expansion',0) < 0.8)
        elif regime == Regime.TEND_BAIXA:
            base -= 0.2*max(0,momentum)
        return ScoreBreakdown(total=base, momentum=momentum, breakout=breakout, contrarian=contrarian, penalty=overext)
    
    updated_scoring_py = """from typing import Dict, Any, NamedTuple
from core.regimes import Regime

class ScoreBreakdown(NamedTuple):
    total: float
    momentum: float
    breakout: float
    contrarian: float
    penalty: float
    extras: Dict[str, float]

class ScoringEngine:
    def __init__(self, weights: Dict[str,float]):
        self.w = weights

    def compute(self, row: Dict[str, Any], regime: Regime) -> ScoreBreakdown:
        momentum = (row.get('ret_4',0) + 0.5*row.get('ret_24',0)) * row.get('mom_quality',1)
        breakout = 0.0
        if row.get('above_pivot_high') and row.get('range_expansion',0) > 1.2:
            breakout = row.get('range_expansion',0)
        contrarian = 0.0
        if 'funding_z' in row and 'ret_4' in row:
            # Exemplo: funding muito positivo mas preço anda pouco -> sinal de esgotamento
            if row['funding_z'] > 2 and abs(row['ret_4']) < 0.002:
                contrarian = 0.02 * row['funding_z']
        if 'oi_delta_pct' in row and row['oi_delta_pct'] < -0.05 and row.get('ret_24',0) > 0:
            contrarian += 0.01  # retirada de OI durante alta pode indicar short covering (fragilidade)
        overext = max(0, row.get('overext_z',0) - 2)
        base = (self.w['momentum_w']*momentum + self.w['breakout_w']*breakout + self.w['contrarian_w']*contrarian) - 0.05*overext
        # Ajuste por regime
        if regime == Regime.VOL_COMPRESSA:
            base += (1 - min(1, row.get('rci_20',1))) * 0.05  # recompensa compressão
        elif regime == Regime.TEND_BAIXA:
            base -= max(0, momentum)
        extras = {
            'rci_20': row.get('rci_20',0),
            'overext_z': row.get('overext_z',0)
        }
        return ScoreBreakdown(total=base, momentum=momentum, breakout=breakout, contrarian=contrarian, penalty=overext, extras=extras)
"""