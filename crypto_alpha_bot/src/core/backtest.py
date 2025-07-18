import pandas as pd
from typing import Dict, List
from core.scoring import ScoringEngine
from core.regimes import RegimeClassifier
from statistics import pstdev, fmean

class MultiAssetBacktester:
    def __init__(self, hold_bars: int = 6, fee_bps: float = 7):
        self.hold = hold_bars
        self.fee = fee_bps / 10000.0  # one-way

    def run(self, data_map: Dict[str, pd.DataFrame], scorer: ScoringEngine, regime_clf: RegimeClassifier):
        equity = 1.0
        trades = []
        trade_log = []
        for symbol, df in data_map.items():
            if len(df) < 120:
                continue
            for i in range(120, len(df) - self.hold):
                window = df.iloc[: i + 1]
                regime = regime_clf.classify(window)
                row = window.iloc[-1].to_dict()
                sb = scorer.compute(row, regime)
                if sb.total > 0.05:  # gatilho simples
                    entry = row["close"]
                    exit_price = df.iloc[i + self.hold]["close"]
                    pnl = (exit_price / entry) - 1 - 2 * self.fee
                    equity *= (1 + pnl * 0.2)  # arrisca 20% do capital (placeholder)
                    trades.append(pnl)
                    trade_log.append(
                        {
                            "symbol": symbol,
                            "i": i,
                            "entry": entry,
                            "exit": exit_price,
                            "pnl": pnl,
                            "score": sb.total,
                            "regime": regime,
                        }
                    )
        metrics = self._metrics(trades, equity)
        return {"equity": equity, "trades": trades, "trade_log": trade_log, "metrics": metrics}

    def _metrics(self, trades: List[float], equity: float):
        if not trades:
            return {"sharpe": 0, "trades": 0, "equity": equity}
        mean = fmean(trades)
        sd = pstdev(trades) or 1e-9
        sharpe = mean / sd
        win_rate = sum(1 for t in trades if t > 0) / len(trades)
        return {
            "sharpe": sharpe,
            "trades": len(trades),
            "win_rate": win_rate,
            "final_equity": equity,
            "avg_pnl": mean,
        }
