import pandas as pd
import logging

LOG = logging.getLogger("integrate")


def _coerce_int(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def merge_derivatives_into_price(df: pd.DataFrame,
                                 funding_df: pd.DataFrame | None,
                                 oi_df: pd.DataFrame | None,
                                 keep_funding_time: bool = False) -> pd.DataFrame:
    """
    Alinha funding (funding_time) e OI (timestamp) aos candles (open_time) via merge_asof,
    com robustez para múltiplas variações de nomes de colunas e conflitos.
    """
    if df is None or df.empty:
        return df

    if "open_time" not in df.columns:
        LOG.warning("DataFrame de preço sem 'open_time'; retorno inalterado.")
        return df

    price = df.copy().sort_values("open_time")
    price = _coerce_int(price, ["open_time"])

        # ---------- FUNDING ----------
    if funding_df is not None and not funding_df.empty:
        f = funding_df.copy()

        # Remove symbol para evitar conflitos
        if "symbol" in f.columns:
            f = f.drop(columns=["symbol"])

        # Se existir funding_rate (ou variações), normaliza para 'funding'
        for cand in ["funding", "funding_rate", "fundingRate", "last_funding_rate", "lastFundingRate", "rate"]:
            if cand in f.columns and cand != "funding":
                f = f.rename(columns={cand: "funding"})
                break

        # Garante que funding_time existe
        if "funding_time" not in f.columns:
            for alt in ("time", "timestamp"):
                if alt in f.columns:
                    f = f.rename(columns={alt: "funding_time"})
                    break

        # Se ainda não temos funding OR funding_time -> aborta
        if "funding" not in f.columns or "funding_time" not in f.columns:
            LOG.warning("Funding merge skip (sem 'funding' ou 'funding_time'): %s", list(f.columns))
        else:
            f = f[["funding_time", "funding"]].drop_duplicates().sort_values("funding_time")
            f = _coerce_int(f, ["funding_time"])
            # Evitar merge duplicado se 'funding' já existir em price: renomeia temporariamente
            if "funding" in price.columns:
                f = f.rename(columns={"funding": "funding_new"})
                funding_col_name = "funding_new"
            else:
                funding_col_name = "funding"

            merged = pd.merge_asof(
                price,
                f,
                left_on="open_time",
                right_on="funding_time",
                direction="backward",
                suffixes=("", "_fund")
            )
            price = merged

            # Se renomeámos para funding_new, combinar
            if funding_col_name == "funding_new":
                # Preenche NaNs antigos com novos e remove coluna auxiliar
                price["funding"] = price["funding"].fillna(price["funding_new"])
                price = price.drop(columns=["funding_new"])

            if not keep_funding_time and "funding_time" in price.columns:
                price = price.drop(columns=["funding_time"])

    # ---------- OPEN INTEREST ----------
    if oi_df is not None and not oi_df.empty:
        o = oi_df.copy()
        if "symbol" in o.columns:
            o = o.drop(columns=["symbol"])
        o = _coerce_int(o, ["timestamp"])
        o = o.sort_values("timestamp")

        dropping = [c for c in o.columns if c != "timestamp" and c in price.columns]
        if dropping:
            o = o.drop(columns=dropping)

        cols_new = [c for c in o.columns if c != "timestamp"]
        if cols_new:
            price = pd.merge_asof(
                price,
                o,
                left_on="open_time",
                right_on="timestamp",
                direction="backward",
                suffixes=("", "_oi")
            )
            if "timestamp" in price.columns:
                price = price.drop(columns=["timestamp"])

    return price
