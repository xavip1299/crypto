import math

def sharpe(trades: List[float], risk_free: float=0.0) -> float:
    if not trades:
        return 0.0
    import statistics
    mean = statistics.fmean(trades)
    sd = statistics.pstdev(trades) or 1e-9
    return (mean - risk_free)/(sd)