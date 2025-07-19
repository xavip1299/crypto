from __future__ import annotations
import yaml, os
from pathlib import Path

CANDIDATES = [
    Path("settings.yaml"),
    Path("config/settings.yaml"),
    Path("config/setting.yaml"),
]

def load_settings():
    for p in CANDIDATES:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"Nenhum settings.yaml encontrado nas opções: {[str(c) for c in CANDIDATES]}")
