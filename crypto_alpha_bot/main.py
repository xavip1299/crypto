main_py_example = """#!/usr/bin/env python
import asyncio, yaml, pandas as pd
from core.datasource import DataSourceManager
from core.features import FeatureEngineer
from core.regimes import RegimeClassifier
from core.scoring import ScoringEngine
from core.risk import RiskManager
from core.backtest import SimpleBacktester
from core.metrics import sharpe

async def collect_ohlcv(symbol: str, dsm: DataSourceManager, limit: int=200):
    # Para demo: apenas 1h -> gerar dataset artificial posteriormente
    data = await dsm.fetch_binance_klines(symbol, '1h', limit=limit)
    return pd.DataFrame(data)

async def main():
    settings = yaml.safe_load(open('config/settings.yaml'))
    weights = settings['scoring']
    fe = FeatureEngineer()
    regime_clf = RegimeClassifier()
    scoring = ScoringEngine(weights)
    backtester = SimpleBacktester()

    async with DataSourceManager(settings) as dsm:
        df = await collect_ohlcv('BTCUSDT', dsm)
    df = fe.basic_features(df)
    df = fe.add_zscores(df, ['ret_1','ret_24'])
    result = backtester.run(df, scoring, regime_clf)
    print('Equity:', result['equity'])
    print('Sharpe:', sharpe(result['trades']))

if __name__ == '__main__':
    asyncio.run(main())
"""

main_usage_snippet = """import asyncio, pandas as pd
from core.datasource import fetch_klines
from core.features import build_feature_set
from core.regimes import RegimeClassifier
from core.scoring import ScoringEngine

async def run():
    raw = await fetch_klines('BTCUSDT','1h',400)
    df = pd.DataFrame(raw)
    df = build_feature_set(df)
    clf = RegimeClassifier()
    regime = clf.classify(df)
    scorer = ScoringEngine({'momentum_w':0.4,'breakout_w':0.3,'contrarian_w':0.3})
    last = df.iloc[-1].to_dict()
    score = scorer.compute(last, regime)
    print('Regime:', regime)
    print('Score breakdown:', score)

if __name__ == '__main__':
    asyncio.run(run())
"""