from core.config_loader import load_settings
import glob, os, pandas as pd

st = load_settings()
spot = st["universe"]["spot"]
perps = st["universe"]["perps"]
print("SETTINGS spot:", spot)
print("SETTINGS perps:", perps)

raw_paths = glob.glob("data/raw/*.parquet")
print("N raw parquet:", len(raw_paths))

symbols_from_raw = set()
for p in raw_paths:
    base = os.path.basename(p)
    for s in spot:
        if s in base:
            symbols_from_raw.add(s)
print("Símbolos detectados em raw:", sorted(symbols_from_raw))

def info(p):
    if os.path.exists(p):
        df = pd.read_parquet(p)
        print(f"{p}: symbols={df.symbol.nunique()} sample={sorted(df.symbol.unique())[:10]} linhas={len(df)}")
    else:
        print(f"{p}: (não existe)")

info("data/reports/historical_signals.parquet")
info("data/reports/historical_signals_enriched.parquet")
