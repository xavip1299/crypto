import pandas as pd

class SimpleBacktester:
    def __init__(self, fee_bps: float = 7):
        self.fee = fee_bps/10000.0

    def run(self, df: pd.DataFrame, scoring_engine: ScoringEngine, regime_clf: RegimeClassifier):
        equity = 1.0
        trades = []
        df = df.copy()
        for i in range(60, len(df)):
            window = df.iloc[:i]
            regime = regime_clf.classify(window)
            row = window.iloc[-1].to_dict()
            score = scoring_engine.compute(row, regime)
            if score.total > 0.05:  # gatilho simples
                # entrar e sair após N barras ou alvo/stop (simplificado)
                entry_price = row['close']
                exit_idx = min(i+6, len(df)-1)
                exit_price = df.iloc[exit_idx]['close']
                pnl = (exit_price/entry_price - 1) - self.fee*2
                equity *= (1 + pnl*0.2)  # arriscar 20% do capital por simplicidade
                trades.append(pnl)
        return {"equity": equity, "trades": trades}