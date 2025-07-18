from __future__ import annotations
import json, time, math, logging, pathlib, statistics, asyncio, hashlib, random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable

LOG = logging.getLogger("crypto_alpha_bot")

DATA_DIR = pathlib.Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROC_DIR = DATA_DIR / "processed"
LOGS_DIR = pathlib.Path("logs")
REPORTS_DIR = pathlib.Path("reports")

for _d in [DATA_DIR, RAW_DIR, INTERIM_DIR, PROC_DIR, LOGS_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def now_ts() -> float:
    return time.time()


def sha1_obj(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def ewma(values: List[float], alpha: float) -> List[float]:
    out = []
    prev = None
    for v in values:
        prev = v if prev is None else alpha * v + (1 - alpha) * prev
        out.append(prev)
    return out


def zscore(series: List[float]) -> List[float]:
    if len(series) < 2:
        return [0]*len(series)
    mean = statistics.fmean(series)
    stdev = statistics.pstdev(series) or 1e-9
    return [(x-mean)/stdev for x in series]
