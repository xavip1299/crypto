import aiohttp
import logging
from typing import Optional

LOG = logging.getLogger("alerts")

class TelegramAlerter:
    """Envio de mensagens para Telegram. Usa modo fire-and-forget básico."""

    def __init__(self, bot_token: str, chat_id: str | int, parse_mode: str = "Markdown"):
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.chat_id = chat_id
        self.parse_mode = parse_mode

    async def send(self, text: str) -> bool:
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": self.parse_mode,
            "disable_web_page_preview": True,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as r:
                    if r.status != 200:
                        LOG.warning("Falha Telegram (%s): %s", r.status, await r.text())
                        return False
                    return True
        except Exception as e:
            LOG.warning("Erro Telegram: %s", e)
            return False
