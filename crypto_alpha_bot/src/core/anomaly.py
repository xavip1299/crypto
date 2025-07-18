class AnomalyDetector:
    def __init__(self):
        self.consecutive = 0

    def check(self, series: List[float]) -> bool:
        # EWMA + desvio simples
        if len(series) < 30:
            return False
        tail = series[-1]
        mean = statistics.fmean(series[-30:])
        st = statistics.pstdev(series[-30:]) or 1e-9
        if abs(tail-mean) > 5*st:
            self.consecutive += 1
            return True
        self.consecutive = 0
        return False
