"""
Composite scoring model: insider, volume, price structure, catalyst.
Pure functions — applies formulas row-wise and returns an enriched DataFrame.
"""

import logging

import pandas as pd

from src.config import (
    GRADE_A,
    GRADE_A_PLUS,
    GRADE_B,
    INSIDER_BUY_7D_HIGH,
    INSIDER_BUY_7D_LOW,
    INSIDER_BUY_7D_MID,
    INSIDER_BUY_MKTCAP_BONUS,
    INSIDER_BUY_MKTCAP_THRESHOLD,
    INSIDER_CEO_CFO_BONUS,
    INSIDER_CLUSTER_BONUS,
    WEIGHT_CATALYST,
    WEIGHT_INSIDER,
    WEIGHT_PRICE,
    WEIGHT_VOLUME,
)

logger = logging.getLogger(__name__)


def compute_scores(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add insider_score, volume_score, price_structure_score, catalyst_score,
    composite_score, and grade to a copy of features_df.
    """
    df = features_df.copy()

    df["insider_score"]          = df.apply(_score_insider, axis=1)
    df["volume_score"]           = df.apply(_score_volume, axis=1)
    df["price_structure_score"]  = df.apply(_score_price_structure, axis=1)
    df["catalyst_score"]         = 0.0  # Phase 1 stub

    df["composite_score"] = (
        df["insider_score"]         * WEIGHT_INSIDER
        + df["volume_score"]        * WEIGHT_VOLUME
        + df["price_structure_score"] * WEIGHT_PRICE
        + df["catalyst_score"]      * WEIGHT_CATALYST
    ).round(1)

    df["grade"] = df["composite_score"].apply(_assign_grade)

    logger.debug(
        "compute_scores: A+=%d  A=%d  B=%d  C=%d",
        (df["grade"] == "A+").sum(),
        (df["grade"] == "A").sum(),
        (df["grade"] == "B").sum(),
        (df["grade"] == "C").sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Row-level scorers
# ---------------------------------------------------------------------------

def _score_insider(row: pd.Series) -> float:
    pts = 0

    buy_7d = float(row.get("insider_buy_7d", 0))
    if buy_7d > INSIDER_BUY_7D_HIGH:
        pts += 40
    elif buy_7d > INSIDER_BUY_7D_MID:
        pts += 25
    elif buy_7d > INSIDER_BUY_7D_LOW:
        pts += 15

    if row.get("insider_cluster", False):
        pts += INSIDER_CLUSTER_BONUS
    if row.get("has_ceo_cfo_buy", False):
        pts += INSIDER_CEO_CFO_BONUS
    if float(row.get("buy_vs_market_cap_ratio", 0)) > INSIDER_BUY_MKTCAP_THRESHOLD:
        pts += INSIDER_BUY_MKTCAP_BONUS

    return _clamp(float(pts))


def _score_volume(row: pd.Series) -> float:
    pts = 0
    ratio = float(row.get("volume_ratio_5d", 1.0))

    if ratio > 3.0:
        pts += 40
    elif ratio > 2.0:
        pts += 30
    elif ratio > 1.5:
        pts += 20
    elif ratio > 1.3:
        pts += 10

    if row.get("volume_spike", False):
        pts += 30
    if row.get("accumulation_volume", False):
        pts += 30

    return _clamp(float(pts))


def _score_price_structure(row: pd.Series) -> float:
    pts = 0
    price_range = float(row.get("price_range_20d", 100.0))
    price_chg   = float(row.get("price_change_5d", 0.0))

    if row.get("near_breakout", False):
        pts += 40

    if price_range < 10.0:
        pts += 30
    elif price_range < 15.0:
        pts += 20

    if price_chg > 5.0:
        pts += 15
    elif price_chg > 0.0:
        pts += 10

    if row.get("post_drop", False):
        pts -= 20

    return _clamp(float(pts))


def _assign_grade(composite: float) -> str:
    if composite >= GRADE_A_PLUS:
        return "A+"
    if composite >= GRADE_A:
        return "A"
    if composite >= GRADE_B:
        return "B"
    return "C"


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))
