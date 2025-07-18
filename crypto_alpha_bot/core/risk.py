class RiskManager:
    def __init__(self, settings: Dict[str, Any]):
        self.capital = settings['risk']['capital_usd']
        self.max_risk_pct_trade = settings['risk']['max_risk_pct_trade']/100
        self.daily_loss_limit_pct = settings['risk']['max_daily_loss_pct']/100
        self.max_total_exposure_pct = settings['risk']['max_total_exposure_pct']/100
        self.current_daily_pnl = 0.0
        self.positions = {}

    def size_position(self, symbol: str, stop_dist: float, price: float, confidence: float = 1.0) -> float:
        risk_amount = self.capital * self.max_risk_pct_trade * confidence
        qty = risk_amount / max(stop_dist, 1e-9)
        notional = qty * price
        return min(notional, self.capital * 0.1)  # limiter adicional

    def can_open(self, symbol: str) -> bool:
        # TODO: adicionar checks de correlação, exposure sectorial
        return self.current_daily_pnl > - self.capital * self.daily_loss_limit_pct