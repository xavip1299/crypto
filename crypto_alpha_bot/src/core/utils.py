import logging, json, time, pathlib, hashlib
from typing import Any, Dict, List
from dataclasses import dataclass

def setup_logging(level: str = "INFO"):
    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord):
            base = {
                "ts": record.created,
                "level": record.levelname,
                "msg": record.getMessage(),
                "module": record.module,
                "name": record.name,
            }
            return json.dumps(base)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler])

def sha1_obj(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

DATA_RAW = pathlib.Path("data/raw")
DATA_RAW.mkdir(parents=True, exist_ok=True)

def now_ts() -> float:
    return time.time()

@dataclass
class SignalResult:
    symbol: str
    timestamp: int
    score: float
    regime: str
    momentum: float
    breakout: float
    contrarian: float
    penalty: float
