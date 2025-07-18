unit_test_example = """import pandas as pd
from core.scoring import ScoringEngine
from core.regimes import Regime

def test_scoring_basic():
    scoring = ScoringEngine({'momentum_w':0.4,'breakout_w':0.3,'contrarian_w':0.3})
    row = {'ret_4':0.01,'ret_24':0.02,'range_expansion':1.5}
    res = scoring.compute(row, Regime.TEND_ALTA)
    assert res.total != 0
"""