class PortfolioState:
    def __init__(self):
        self.positions: Dict[str, Dict[str, Any]] = {}

    def update(self, symbol: str, notional: float, side: str):
        self.positions[symbol] = {"notional": notional, "side": side}