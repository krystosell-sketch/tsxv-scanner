"""
Feature engineering: one row per ticker containing all trading signals.
Pure functions — no side effects.
"""

import logging

import numpy as np
import pandas as pd

from src.config import (
    ACCUM_VOLUME_MIN_DAYS_ABOVE_AVG,
    CLUSTER_MIN_INSIDERS,
    CLUSTER_WINDOW_DAYS,
    NEAR_BREAKOUT_THRESHOLD,
    POST_DROP_THRESHOLD,
    ROLE_WEIGHTS,
    VOLUME_SPIKE_MULTIPLIER,
)

logger = logging.getLogger(__name__)


def compute_features(
    insider_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    market_caps: dict[str, float],
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Master function. Returns features_df with one row per ticker in ohlcv_df.
    Tickers with no insider activity get insider features = 0 / False / NaT.
    reference_date defaults to the max trade_date in insider_df.
    """
    if reference_date is None:
        if not insider_df.empty and "trade_date" in insider_df.columns:
            reference_date = insider_df["trade_date"].max()
        else:
            reference_date = pd.Timestamp.now().normalize()

    logger.debug("compute_features reference_date=%s", reference_date)

    # Build ticker universe from ohlcv
    tickers = pd.Series(ohlcv_df["ticker"].unique(), name="ticker").to_frame()

    insider_feat = _compute_insider_features(insider_df, reference_date)
    volume_feat  = _compute_volume_features(ohlcv_df)
    price_feat   = _compute_price_features(ohlcv_df)

    # Merge — left join keeps all ohlcv tickers
    df = tickers.merge(insider_feat, on="ticker", how="left")
    df = df.merge(volume_feat, on="ticker", how="left")
    df = df.merge(price_feat, on="ticker", how="left")

    # Fill missing insider columns with defaults
    _fill_insider_defaults(df)

    # buy_vs_market_cap_ratio
    df["buy_vs_market_cap_ratio"] = _compute_buy_vs_mktcap(df, market_caps)

    logger.debug("compute_features: %d tickers, %d columns", len(df), len(df.columns))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sub-feature computers
# ---------------------------------------------------------------------------

def _compute_insider_features(
    insider_df: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """Per-ticker insider aggregations."""
    if insider_df.empty:
        return pd.DataFrame(columns=[
            "ticker", "insider_buy_7d", "insider_buy_30d", "insider_buy_count_7d",
            "insider_cluster", "insider_role_weight", "delta_ownership_pct",
            "has_ceo_cfo_buy", "latest_trade_date", "insider_name", "latest_role",
            "latest_value_cad", "issuer",
        ])

    cutoff_7d  = reference_date - pd.Timedelta(days=7)
    cutoff_14d = reference_date - pd.Timedelta(days=CLUSTER_WINDOW_DAYS)
    cutoff_30d = reference_date - pd.Timedelta(days=30)

    results: list[dict] = []

    for ticker, grp in insider_df.groupby("ticker"):
        grp7  = grp[grp["trade_date"] >= cutoff_7d]
        grp14 = grp[grp["trade_date"] >= cutoff_14d]
        grp30 = grp[grp["trade_date"] >= cutoff_30d]

        # Latest trade (most recent overall in 30d)
        latest = grp30.sort_values("trade_date").iloc[-1] if not grp30.empty else grp.sort_values("trade_date").iloc[-1]

        distinct_insiders_14d = grp14["insider_name"].nunique()
        cluster = distinct_insiders_14d >= CLUSTER_MIN_INSIDERS

        role_weights_14d = grp14["role"].map(ROLE_WEIGHTS).fillna(1)
        max_role_weight = float(role_weights_14d.max()) if not role_weights_14d.empty else 1.0

        has_ceo_cfo = grp14["role"].isin(["CEO", "CFO"]).any()

        results.append({
            "ticker":              ticker,
            "insider_buy_7d":      float(grp7["value_cad"].sum()),
            "insider_buy_30d":     float(grp30["value_cad"].sum()),
            "insider_buy_count_7d": int(len(grp7)),
            "insider_cluster":     bool(cluster),
            "insider_role_weight": max_role_weight,
            "delta_ownership_pct": float(grp30["delta_own_pct"].max()) if not grp30.empty else 0.0,
            "has_ceo_cfo_buy":     bool(has_ceo_cfo),
            "latest_trade_date":   latest["trade_date"],
            "insider_name":        latest["insider_name"],
            "latest_role":         latest["role"],
            "latest_value_cad":    float(latest["value_cad"]),
            "issuer":              str(latest.get("issuer", "") or ""),
        })

    return pd.DataFrame(results)


def _compute_volume_features(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker volume features using the most recent 20 rows."""
    results: list[dict] = []

    for ticker, grp in ohlcv_df.groupby("ticker"):
        grp = grp.sort_values("date").tail(20)
        vols = grp["volume"].values.astype(float)

        avg_20d = float(np.mean(vols)) if len(vols) > 0 else 1.0
        avg_5d  = float(np.mean(vols[-5:])) if len(vols) >= 5 else avg_20d

        ratio = avg_5d / avg_20d if avg_20d > 0 else 1.0
        spike = ratio >= VOLUME_SPIKE_MULTIPLIER

        # Accumulation volume: how many of last 5 days are above the 20d avg
        last5 = vols[-5:] if len(vols) >= 5 else vols
        days_above = int(np.sum(last5 > avg_20d))
        accum_vol = days_above >= ACCUM_VOLUME_MIN_DAYS_ABOVE_AVG

        results.append({
            "ticker":              ticker,
            "volume_ratio_5d":     round(ratio, 4),
            "volume_spike":        bool(spike),
            "accumulation_volume": bool(accum_vol),
        })

    return pd.DataFrame(results)


def _compute_price_features(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker price structure features."""
    results: list[dict] = []

    for ticker, grp in ohlcv_df.groupby("ticker"):
        grp = grp.sort_values("date")
        closes = grp["close"].values.astype(float)
        highs  = grp["high"].values.astype(float)
        lows   = grp["low"].values.astype(float)

        last_close = closes[-1]
        high_20d   = float(np.max(highs[-20:])) if len(highs) >= 1 else last_close
        low_20d    = float(np.min(lows[-20:]))  if len(lows) >= 1 else last_close

        # Price range 20d as percentage of last close (uses full intraday range)
        price_range_20d = (high_20d - low_20d) / last_close * 100 if last_close > 0 else 0.0

        # Price change 5d
        close_5d_ago = closes[-5] if len(closes) >= 5 else closes[0]
        price_change_5d = (last_close - close_5d_ago) / close_5d_ago * 100 if close_5d_ago > 0 else 0.0

        # Near breakout: close >= close_low_20d + 90% of close range
        # Uses closing prices so the signal fires when the stock closes near its range high
        close_high_20d = float(np.max(closes[-20:])) if len(closes) >= 1 else last_close
        close_low_20d  = float(np.min(closes[-20:])) if len(closes) >= 1 else last_close
        close_range    = close_high_20d - close_low_20d
        if close_range > 0:
            near_breakout = last_close >= (close_low_20d + NEAR_BREAKOUT_THRESHOLD * close_range)
        else:
            near_breakout = False

        # Post drop: last close < 30-day-ago close * (1 - threshold)
        close_30d_ago = closes[0] if len(closes) >= 30 else closes[0]
        post_drop = last_close < close_30d_ago * (1 - POST_DROP_THRESHOLD)

        results.append({
            "ticker":          ticker,
            "price_range_20d": round(price_range_20d, 2),
            "price_change_5d": round(price_change_5d, 2),
            "near_breakout":   bool(near_breakout),
            "post_drop":       bool(post_drop),
        })

    return pd.DataFrame(results)


def _compute_buy_vs_mktcap(
    df: pd.DataFrame,
    market_caps: dict[str, float],
) -> pd.Series:
    """insider_buy_30d / market_cap_cad, returns Series indexed by df.index."""
    def ratio(row: pd.Series) -> float:
        mktcap = market_caps.get(row["ticker"], 0.0)
        if mktcap <= 0:
            return 0.0
        return float(row.get("insider_buy_30d", 0.0)) / mktcap

    return df.apply(ratio, axis=1)


def _fill_insider_defaults(df: pd.DataFrame) -> None:
    """Fill NaN insider columns with sensible defaults (in-place)."""
    float_cols = ["insider_buy_7d", "insider_buy_30d", "delta_ownership_pct",
                  "insider_role_weight", "latest_value_cad", "buy_vs_market_cap_ratio"]
    int_cols   = ["insider_buy_count_7d"]
    bool_cols  = ["insider_cluster", "has_ceo_cfo_buy"]
    str_cols   = ["insider_name", "latest_role"]

    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")
