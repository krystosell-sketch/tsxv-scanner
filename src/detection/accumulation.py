"""
Signal detection: accumulation, pre-breakout, cluster buying.
Adds boolean flags and human-readable signal_reasons to scored_df.
Pure functions.
"""

import logging

import pandas as pd

from src.config import (
    ACCUM_MAX_PRICE_RANGE_20D_PCT,
    ACCUM_MIN_BUY_7D_CAD,
    ACCUM_MIN_BUY_COUNT_7D,
    ACCUM_MIN_VOLUME_RATIO_5D,
    CLUSTER_MIN_INSIDERS,
)

logger = logging.getLogger(__name__)


def detect_signals(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add accumulation_flag, pre_breakout_flag, cluster_buying_flag,
    and signal_reasons columns to a copy of scored_df.
    """
    df = scored_df.copy()

    df["accumulation_flag"]   = _apply_accumulation_flag(df)
    df["pre_breakout_flag"]   = _apply_pre_breakout_flag(df)
    df["cluster_buying_flag"] = _apply_cluster_buying_flag(df)
    df["signal_reasons"]      = df.apply(_build_signal_reasons, axis=1)

    n_accum  = df["accumulation_flag"].sum()
    n_pre    = df["pre_breakout_flag"].sum()
    n_clst   = df["cluster_buying_flag"].sum()
    logger.debug("detect_signals: ACCUM=%d  PRE-BRK=%d  CLST=%d", n_accum, n_pre, n_clst)

    return df


# ---------------------------------------------------------------------------
# Vectorised flag computers
# ---------------------------------------------------------------------------

def _apply_accumulation_flag(df: pd.DataFrame) -> pd.Series:
    return (
        (df["insider_buy_7d"]      > ACCUM_MIN_BUY_7D_CAD)
        & (df["insider_buy_count_7d"] >= ACCUM_MIN_BUY_COUNT_7D)
        & (df["volume_ratio_5d"]   > ACCUM_MIN_VOLUME_RATIO_5D)
        & (df["price_range_20d"]   < ACCUM_MAX_PRICE_RANGE_20D_PCT)
    )


def _apply_pre_breakout_flag(df: pd.DataFrame) -> pd.Series:
    return (
        df["accumulation_flag"]
        & df["volume_spike"]
        & df["near_breakout"]
    )


def _apply_cluster_buying_flag(df: pd.DataFrame) -> pd.Series:
    return (
        df["insider_cluster"]
        & (df["insider_buy_count_7d"] >= CLUSTER_MIN_INSIDERS)
    )


# ---------------------------------------------------------------------------
# Signal reason builder
# ---------------------------------------------------------------------------

def _build_signal_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []

    buy_7d = float(row.get("insider_buy_7d", 0))
    if buy_7d > 0:
        role = row.get("latest_role", "")
        role_str = f" ({role})" if role else ""
        reasons.append(f"Insider buy ${buy_7d:,.0f} in 7d{role_str}")

    if row.get("insider_cluster", False):
        count = int(row.get("insider_buy_count_7d", 0))
        reasons.append(f"Cluster: {count}+ transactions in 14d (multi-insider)")

    if row.get("has_ceo_cfo_buy", False):
        reasons.append("CEO/CFO buying")

    ratio = float(row.get("buy_vs_market_cap_ratio", 0))
    if ratio > 0.01:
        reasons.append(f"Buy/mktcap ratio {ratio*100:.1f}% (very strong signal)")

    vol_ratio = float(row.get("volume_ratio_5d", 1.0))
    if vol_ratio > 1.3:
        reasons.append(f"Volume {vol_ratio:.1f}x 20d avg")

    if row.get("volume_spike", False):
        reasons.append("Volume spike >2x 20d avg")

    if row.get("accumulation_volume", False):
        reasons.append("Progressive volume accumulation (3+ days above avg)")

    price_range = float(row.get("price_range_20d", 100))
    if price_range < 15.0:
        reasons.append(f"Price in tight base ({price_range:.1f}% range)")

    if row.get("near_breakout", False):
        reasons.append("Near 20d breakout level (>90% of range)")

    chg5d = float(row.get("price_change_5d", 0))
    if chg5d > 5.0:
        reasons.append(f"Upward momentum +{chg5d:.1f}% in 5 days")

    if row.get("post_drop", False):
        reasons.append("Post-drop setup (>20% decline from 30d high)")

    return reasons
