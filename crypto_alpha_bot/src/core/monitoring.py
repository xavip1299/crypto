class HealthSnapshot(NamedTuple):
    ts: float
    api_fail_rate_pct: float
    data_freshness_ok: bool
    anomaly_flags: int
    drawdown_pct: float

class Monitoring:
    def __init__(self):
        self.snapshots: List[HealthSnapshot] = []

    def record(self, snap: HealthSnapshot):
        self.snapshots.append(snap)

    def latest(self) -> Optional[HealthSnapshot]:
        return self.snapshots[-1] if self.snapshots else None
