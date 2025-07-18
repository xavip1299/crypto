import yaml, pandas as pd
from core.storage import ParquetStore
from core.features import build_feature_set
from core.regimes import RegimeClassifier
from core.scoring import ScoringEngine
from core.backtest import MultiAssetBacktester

def load_featured(symbols, settings):
    data_map = {}
    for sym in symbols:
        store = ParquetStore(settings["storage"]["path"], sym, settings["data"]["timeframe"])
        df = store.load()
        if df.empty:
            continue
        df = build_feature_set(df)
        data_map[sym] = df
    return data_map

def main():
    settings = yaml.safe_load(open("config/settings.yaml"))
    symbols = settings["universe"]["spot"]
    data_map = load_featured(symbols, settings)

    clf = RegimeClassifier(
        vol_short_window=settings["regime"]["vol_short_window"],
        vol_long_window=settings["regime"]["vol_long_window"],
        compress_ratio=settings["regime"]["compress_ratio"],
        expansion_ratio=settings["regime"]["expansion_ratio"],
    )
    scorer = ScoringEngine(settings["scoring"])
    bt = MultiAssetBacktester(
        hold_bars=settings["backtest"]["hold_bars"],
        fee_bps=settings["backtest"]["fee_bps"],
    )
    result = bt.run(data_map, scorer, clf)
    print("Backtest metrics:", result["metrics"])
    print("Total trades:", result["metrics"]["trades"])

if __name__ == "__main__":
    main()
