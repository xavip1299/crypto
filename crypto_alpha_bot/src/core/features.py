import pandas as pd
import numpy as np

def pct_change_n(series: pd.Series, n: int) -> pd.Series:
    return series.pct_change(n)

def realized_vol(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).std() * np.sqrt(window)

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()

def zscore(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window)
    return (series - rolling.mean()) / (rolling.std() + 1e-9)

def range_contraction_index(df: pd.DataFrame, window: int = 20) -> pd.Series:
    true_range = (df["high"] - df["low"])
    return true_range / (true_range.rolling(window).mean() + 1e-9)

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1"] = df["close"].pct_change()
    df["ret_4"] = pct_change_n(df["close"], 4)
    df["ret_24"] = pct_change_n(df["close"], 24)
    df["vol_real_24"] = realized_vol(df["ret_1"], 24)
    df["vol_real_72"] = realized_vol(df["ret_1"], 72)
    df["atr"] = atr(df)
    df["range_expansion"] = (df["high"] - df["low"]) / (df["atr"] + 1e-9)
    df["rci_20"] = range_contraction_index(df, 20)
    df["overext_z"] = zscore(df["ret_24"], 30)
    return df

def add_breakout_levels(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["pivot_high"] = df["high"].rolling(lookback).max().shift(1)
    df["pivot_low"] = df["low"].rolling(lookback).min().shift(1)
    df["above_pivot_high"] = df["close"] > df["pivot_high"]
    df["below_pivot_low"] = df["close"] < df["pivot_low"]
    return df

def add_momentum_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mom_quality"] = df["ret_4"] / (df["vol_real_24"] + 1e-9)
    return df

def integrate_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    if "funding" in df.columns:
        df["funding_z"] = zscore(df["funding"], 30)
    if "open_interest_usd" in df.columns:
        df["oi_delta_pct"] = df["open_interest_usd"].pct_change()
    return df

def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    df = add_basic_features(df)
    df = add_breakout_levels(df)
    df = add_momentum_quality(df)
    df = integrate_derivatives(df)
    return df

def recompute_derivative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcula funding_z e oi_delta_pct após merge histórico.
    """
    if "funding" in df.columns:
        df["funding_z"] = zscore(df["funding"], 60)
    if "open_interest_usd" in df.columns:
        df["oi_delta_pct"] = df["open_interest_usd"].pct_change()
    return df
