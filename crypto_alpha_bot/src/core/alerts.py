class AlertManager:
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings

    async def send(self, message: str):
        LOG.info(f"ALERT: {message}")