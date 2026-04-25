"""
Normalize raw DataFrames from ingestion. Pure functions — no mutation.
"""

import logging
import re

import pandas as pd
from dateutil import parser as dateutil_parser

from src.config import ROLE_WEIGHTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role normalization map
# ---------------------------------------------------------------------------
_ROLE_MAP: list[tuple[list[str], str]] = [
    (["chief executive officer", "president and ceo", "president & ceo",
      "president/ceo", "ceo", "president"], "CEO"),
    (["chief financial officer", "vp finance", "vp of finance",
      "vice president finance", "cfo"], "CFO"),
    (["director", "board member", "independent director",
      "non-executive director"], "Director"),
    (["10% holder", "10%+ holder", "10%holder", "major shareholder",
      "significant shareholder"], "10%Holder"),
]


def normalize_ticker(ticker: str) -> str:
    """Uppercase and ensure suffix is exactly '.V' or '.CN'."""
    t = ticker.strip().upper()
    # Normalise common suffix variants
    t = re.sub(r"\.(VAN|TSXV|V)$", ".V", t)
    t = re.sub(r"\.(CN|CSE|NEO)$", ".CN", t)
    return t


def _normalize_role(role: str) -> str:
    """Map free-text role to a canonical ROLE_WEIGHTS key."""
    clean = role.strip().lower()
    for keywords, canonical in _ROLE_MAP:
        if any(kw in clean for kw in keywords):
            return canonical
    return "Other"


def _parse_date_column(series: pd.Series) -> pd.Series:
    """Flexible date parser; returns datetime64[ns]."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(
        series.apply(lambda x: dateutil_parser.parse(str(x)) if pd.notna(x) else pd.NaT),
        utc=False,
        errors="coerce",
    )


def normalize_insider_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean insider trades DataFrame:
    - Parse dates, cast types, normalize roles
    - Keep Buy transactions only; drop option exercises
    - Drop rows with null ticker, null trade_date, or value_cad <= 0
    """
    out = df.copy()

    # Dates
    out["filing_date"] = _parse_date_column(out["filing_date"])
    out["trade_date"]  = _parse_date_column(out["trade_date"])

    # Numerics
    for col in ("price_cad", "value_cad", "delta_own_pct", "market_cap_cad"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    for col in ("quantity", "nb_owned_after"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    # Strings
    for col in ("insider_name", "role", "issuer", "general_remarks", "transaction_type"):
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()

    # Ticker
    out["ticker"] = out["ticker"].astype(str).apply(normalize_ticker)

    # Role normalization
    out["role"] = out["role"].apply(_normalize_role)

    # Filter: buys only
    out = out[out["transaction_type"].str.upper() == "BUY"].copy()

    # Drop option exercises (check general_remarks)
    mask_option = out["general_remarks"].str.lower().str.contains("option exercise", na=False)
    dropped = mask_option.sum()
    if dropped:
        logger.debug("Dropped %d option exercise rows", dropped)
    out = out[~mask_option].copy()

    # Drop invalid rows
    out = out.dropna(subset=["ticker", "trade_date"])
    out = out[out["value_cad"] > 0].copy()

    out = out.reset_index(drop=True)
    logger.debug("normalize_insider_df: %d rows after cleaning", len(out))
    return out


def normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean OHLCV DataFrame:
    - Parse dates, cast types
    - Sort by (ticker, date) ascending
    - Drop rows with close <= 0 or volume < 0
    """
    out = df.copy()

    out["date"]   = _parse_date_column(out["date"])
    out["ticker"] = out["ticker"].astype(str).apply(normalize_ticker)

    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype(int)

    out = out[(out["close"] > 0) & (out["volume"] >= 0)].copy()
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    logger.debug("normalize_ohlcv_df: %d rows after cleaning", len(out))
    return out
