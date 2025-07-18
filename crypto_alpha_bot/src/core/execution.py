class ExecutionEngine:
    def __init__(self):
        pass

    async def send_order(self, symbol: str, side: str, qty: float, price: Optional[float]=None, order_type: str="MARKET"):
        # Placeholder: integrar com a API (modo paper / live)
        LOG.info(f"ORDER {side} {symbol} qty={qty} type={order_type} price={price}")
        return {"id": f"paper-{time.time()}", "symbol": symbol, "side": side, "qty": qty, "price": price}
