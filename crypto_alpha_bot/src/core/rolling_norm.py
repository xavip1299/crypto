from core.rolling_norm import add_symbol_rolling_z
merged = add_symbol_rolling_z(merged, value_col="score", window=120, min_periods=30, out_col="score_z_sym")
