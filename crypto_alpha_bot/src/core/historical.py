import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional
from .datasource import fetch_klines_window

LOG = logging.getLogger("historical")
BINANCE_LIMIT_MAX = 1000


async def paginate_klines(symbol: str,
                          interval: str,
                          total_candles: int,
                          step: int = 1000,
                          end_time_ms: Optional[int] = None) -> List[Dict]:
    """
    Pagina candles históricos para trás (backfill).
    - total_candles: quantos candles no máximo buscar
    - step: por requisição (<= 1000)
    - end_time_ms: timestamp (ms) de referência; se None usa 'agora'

    Retorna lista em ordem cronológica.
    """
    import time

    if step > BINANCE_LIMIT_MAX:
        step = BINANCE_LIMIT_MAX

    if end_time_ms is None:
        end_time_ms = int(time.time() * 1000)

    collected: List[Dict] = []
    remaining = total_candles
    # Estimativa de ms por candle via mapa de intervalos
    ms_per_candle = interval_to_ms(interval)

    async with aiohttp.ClientSession() as session:
        while remaining > 0:
            batch_size = min(step, remaining)
            start_time = end_time_ms - batch_size * ms_per_candle
            rows = await fetch_klines_window(
                session,
                symbol,
                interval,
                limit=batch_size,
                start_time=start_time,
                end_time=end_time_ms
            )
            if not rows:
                LOG.warning("Parando paginação %s: lote vazio.", symbol)
                break
            # Prepend ou acumular e no final ordenar
            collected.extend(rows)
            remaining -= len(rows)
            # Próxima janela termina onde começou a última
            end_time_ms = rows[0]["open_time"]
            # Pequeno respiro para não martelar a API
            await asyncio.sleep(0.25)

            if len(rows) < batch_size:
                # Chegou no início histórico disponível
                break

    # Ordena por open_time
    collected.sort(key=lambda r: r["open_time"])
    return collected


def interval_to_ms(interval: str) -> int:
    """
    Converte string de intervalo Binance em milissegundos.
    Suporta sufixos: m, h, d.
    """
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 60 * 60_000
    if unit == "d":
        return value * 24 * 60 * 60_000
    raise ValueError(f"Intervalo não suportado: {interval}")
