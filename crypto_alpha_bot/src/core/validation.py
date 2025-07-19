from typing import Dict

def validate_settings(cfg: Dict) -> None:
    required = ["universe", "data", "scoring"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Missing required settings key: {k}")