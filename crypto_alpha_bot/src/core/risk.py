class RiskManager:
    def __init__(self, capital_usd: float, max_risk_pct_trade: float):
        self.capital = capital_usd
        self.max_risk_pct_trade = max_risk_pct_trade / 100.0

    def position_size(self, stop_dist: float, price: float, confidence: float = 1.0) -> float:
        risk_amount = self.capital * self.max_risk_pct_trade * confidence
        qty = risk_amount / max(stop_dist, 1e-9)
        notional = qty * price
        return notional
