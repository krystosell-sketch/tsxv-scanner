"""
Generates reproducible, realistic mock data for 20 TSXV/CSE tickers.
Four behavioural profiles ensure all detection paths are exercised.
"""

import logging
import random
from datetime import timedelta

import numpy as np
import pandas as pd

from src.config import MOCK_OHLCV_DAYS, MOCK_TICKERS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ticker profiles
# ---------------------------------------------------------------------------
_STRONG_ACCUM = ["VLE.V", "NVNI.CN", "LITH.V", "PNPN.CN"]
_NEAR_BREAKOUT = ["FIND.V", "CRTM.CN", "AVCR.V", "BLOK.CN"]
_MILD_ACCUM = ["CSTR.V", "GCX.V", "LODE.CN", "MAGT.V", "SAFE.CN"]
_NO_SIGNAL = ["IRON.V", "GLDN.CN", "SLVX.V", "AQUA.CN", "NRTH.V", "PINE.CN", "CBAL.V"]

# Realistic issuer names matching tickers
_ISSUERS: dict[str, str] = {
    "VLE.V": "Vital Energy Corp",
    "NVNI.CN": "Novani Resources Inc",
    "LITH.V": "Lithium Peak Minerals",
    "PNPN.CN": "Pinpoint Mining Ltd",
    "FIND.V": "Find Gold Corp",
    "CRTM.CN": "Criterion Metals Inc",
    "AVCR.V": "Avocier Resources",
    "BLOK.CN": "Blockchain Infrastructure Ltd",
    "CSTR.V": "Coastline Resources",
    "GCX.V": "GoldCreek Explorations",
    "LODE.CN": "Lodestar Minerals Inc",
    "MAGT.V": "Magnetite Capital Corp",
    "SAFE.CN": "SafeHaven Ventures",
    "IRON.V": "Ironwood Mining Corp",
    "GLDN.CN": "Golden Peak Resources",
    "SLVX.V": "Silver Axis Corp",
    "AQUA.CN": "Aqua Clean Energy",
    "NRTH.V": "Northern Shield Inc",
    "PINE.CN": "Pine Ridge Minerals",
    "CBAL.V": "Cobalt Alliance Ltd",
}

# Insider pools per ticker — realistic Canadian names + roles
_INSIDERS: dict[str, list[dict]] = {
    "VLE.V":   [{"name": "Jean-Marc Tremblay", "role": "CEO"},
                {"name": "Sophie Audet",        "role": "CFO"},
                {"name": "Robert Beauchamp",    "role": "Director"}],
    "NVNI.CN": [{"name": "Marie Côté",          "role": "CFO"},
                {"name": "David Huang",         "role": "CEO"},
                {"name": "Linda Park",          "role": "Director"}],
    "LITH.V":  [{"name": "Pierre Lavoie",       "role": "CEO"},
                {"name": "Nathalie Roy",        "role": "Director"},
                {"name": "James Whitmore",      "role": "10%Holder"}],
    "PNPN.CN": [{"name": "Robert Chen",         "role": "10%Holder"},
                {"name": "Andrée Bélanger",     "role": "Director"},
                {"name": "Marc Bouchard",       "role": "CFO"}],
    "FIND.V":  [{"name": "Tom Wilson",          "role": "CEO"},
                {"name": "Sarah Leblanc",       "role": "Director"}],
    "CRTM.CN": [{"name": "Sophie Martin",       "role": "Director"},
                {"name": "André Dubois",        "role": "CEO"}],
    "AVCR.V":  [{"name": "Linda Park",          "role": "CFO"}],
    "BLOK.CN": [{"name": "Kevin Zhang",         "role": "Director"},
                {"name": "Julie Gauthier",      "role": "CEO"}],
    "CSTR.V":  [{"name": "André Dubois",        "role": "Director"}],
    "GCX.V":   [{"name": "Marc Bouchard",       "role": "Director"}],
    "LODE.CN": [{"name": "Tom Wilson",          "role": "CEO"}],
    "MAGT.V":  [{"name": "Caroline Fortin",     "role": "Director"}],
    "SAFE.CN": [{"name": "Eric Champagne",      "role": "Director"}],
    "IRON.V":  [{"name": "Paul Henderson",      "role": "CEO"}],
    "GLDN.CN": [{"name": "Lisa Tremblay",       "role": "Director"}],
    "SLVX.V":  [{"name": "Bob Mackenzie",       "role": "10%Holder"}],
    "AQUA.CN": [{"name": "Diane Pelletier",     "role": "Director"}],
    "NRTH.V":  [{"name": "Chris Ballantyne",    "role": "CEO"}],
    "PINE.CN": [{"name": "Sylvie Morin",        "role": "Director"}],
    "CBAL.V":  [{"name": "Antoine Dupont",      "role": "10%Holder"}],
}

# Base prices (CAD) per ticker — typical penny stock range
_BASE_PRICES: dict[str, float] = {
    "VLE.V": 0.145, "NVNI.CN": 0.320, "LITH.V": 0.190, "PNPN.CN": 0.420,
    "FIND.V": 0.078, "CRTM.CN": 0.055, "AVCR.V": 0.092, "BLOK.CN": 0.310,
    "CSTR.V": 0.165, "GCX.V": 0.230, "LODE.CN": 0.047, "MAGT.V": 0.125,
    "SAFE.CN": 0.088, "IRON.V": 0.034, "GLDN.CN": 0.510, "SLVX.V": 0.072,
    "AQUA.CN": 0.195, "NRTH.V": 0.063, "PINE.CN": 0.041, "CBAL.V": 0.280,
}

# Market caps in CAD
_MARKET_CAPS: dict[str, float] = {
    "VLE.V": 4_500_000,  "NVNI.CN": 8_200_000, "LITH.V": 6_100_000,
    "PNPN.CN": 3_800_000,"FIND.V": 5_500_000,  "CRTM.CN": 2_100_000,
    "AVCR.V": 7_300_000, "BLOK.CN": 12_500_000,"CSTR.V": 9_800_000,
    "GCX.V": 15_000_000, "LODE.CN": 1_900_000, "MAGT.V": 4_200_000,
    "SAFE.CN": 6_700_000,"IRON.V": 3_100_000,  "GLDN.CN": 22_000_000,
    "SLVX.V": 5_400_000, "AQUA.CN": 11_200_000,"NRTH.V": 2_800_000,
    "PINE.CN": 1_600_000,"CBAL.V": 18_500_000,
}


def get_mock_market_caps(tickers: list[str] | None = None) -> dict[str, float]:
    """Returns {ticker: market_cap_cad} for the given tickers (all if None)."""
    t = tickers or MOCK_TICKERS
    return {ticker: _MARKET_CAPS.get(ticker, 5_000_000) for ticker in t}


def generate_ohlcv(
    tickers: list[str] | None = None,
    days: int = MOCK_OHLCV_DAYS,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Returns ohlcv_df with `days` rows per ticker.

    Profile-specific behaviour:
    - Strong accum : tight range (<10%), volume trends up
    - Near-breakout: volume spike last 5 days (>2.5x avg), close near 20d high
    - Mild accum   : moderate range, slight volume uptick
    - No signal    : erratic price, flat/noisy volume
    """
    rng = np.random.default_rng(seed)
    reference_date = pd.Timestamp("2026-04-24")
    t_list = tickers or MOCK_TICKERS
    rows: list[dict] = []

    for ticker in t_list:
        base = _BASE_PRICES.get(ticker, 0.10)
        profile = _profile(ticker)
        dates = [reference_date - timedelta(days=days - 1 - i) for i in range(days)]

        if profile == "strong":
            closes = _gen_tight_closes(rng, base, days, push_top=True)
            volumes = _gen_strong_accum_volume(rng, base_vol=600_000, days=days)
        elif profile == "breakout":
            closes = _gen_near_high_closes(rng, base, days)
            volumes = _gen_spike_volume(rng, base_vol=600_000, days=days)
        elif profile == "mild":
            closes = _gen_moderate_closes(rng, base, days)
            volumes = _gen_flat_volume(rng, base_vol=400_000, days=days, noise=0.4)
        else:  # no signal
            closes = _gen_erratic_closes(rng, base, days)
            volumes = _gen_flat_volume(rng, base_vol=250_000, days=days, noise=0.6)

        # Strong accum tickers get a very tight intraday spread to keep price_range_20d < 10%
        if profile == "strong":
            spread_range = (0.003, 0.012)
        else:
            spread_range = (0.01, 0.04)

        for i, (date, close) in enumerate(zip(dates, closes)):
            spread = close * rng.uniform(*spread_range)
            high = close + spread
            low = max(close - spread, 0.001)
            open_ = low + rng.uniform(0, high - low)
            rows.append({
                "date":   date,
                "ticker": ticker,
                "open":   round(open_, 4),
                "high":   round(high, 4),
                "low":    round(low, 4),
                "close":  round(close, 4),
                "volume": int(volumes[i]),
            })

    df = pd.DataFrame(rows)
    logger.debug("Generated OHLCV: %d rows for %d tickers", len(df), len(t_list))
    return df


def generate_insider_trades(
    tickers: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Returns insider_df with realistic trades matching the four profiles.

    - Strong accum : 3-5 Buy trades in last 7 days by 2+ insiders (CEO+CFO+Dir)
    - Near-breakout: 1-2 Buy trades 10-20 days ago (older signal)
    - Mild accum   : 1-2 Buy trades in last 8-14 days by 1 insider
    - No signal    : Sell or Option Exercise only (never Buy)
    """
    random.seed(seed)
    reference_date = pd.Timestamp("2026-04-24")
    t_list = tickers or MOCK_TICKERS
    rows: list[dict] = []

    for ticker in t_list:
        profile = _profile(ticker)
        insiders = _INSIDERS.get(ticker, [{"name": "Unknown", "role": "Other"}])
        base_price = _BASE_PRICES.get(ticker, 0.10)

        if profile == "strong":
            _add_strong_accum_trades(rows, ticker, insiders, base_price, reference_date)
        elif profile == "breakout":
            _add_breakout_trades(rows, ticker, insiders, base_price, reference_date)
        elif profile == "mild":
            _add_mild_accum_trades(rows, ticker, insiders, base_price, reference_date)
        else:
            _add_no_signal_trades(rows, ticker, insiders, base_price, reference_date)

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "filing_date", "trade_date", "ticker", "issuer", "insider_name",
        "role", "transaction_type", "price_cad", "quantity", "value_cad",
        "nb_owned_after", "delta_own_pct", "general_remarks", "market_cap_cad",
    ])
    logger.debug("Generated %d insider trade rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

def _profile(ticker: str) -> str:
    if ticker in _STRONG_ACCUM:
        return "strong"
    if ticker in _NEAR_BREAKOUT:
        return "breakout"
    if ticker in _MILD_ACCUM:
        return "mild"
    return "none"


# ---------------------------------------------------------------------------
# OHLCV generators
# ---------------------------------------------------------------------------

def _gen_tight_closes(
    rng: np.random.Generator,
    base: float,
    days: int,
    push_top: bool = False,
) -> list[float]:
    """
    Closes in a tight ±6% range around base.
    When push_top=True, the last 5 days drift toward the top of the range
    so near_breakout fires.
    """
    closes = [base]
    for i in range(days - 1):
        if push_top and i >= days - 6:
            # Drift toward base * 1.04 (top of ±4% range)
            target = base * 1.042
            chg = (target - closes[-1]) / closes[-1] * rng.uniform(0.3, 0.7)
        else:
            chg = rng.normal(0.0, 0.006)
            new_tmp = closes[-1] * (1 + chg)
            if abs(new_tmp - base) / base > 0.04:
                chg = -chg * 0.5
        closes.append(max(closes[-1] * (1 + chg), 0.001))
    return closes


def _gen_near_high_closes(rng: np.random.Generator, base: float, days: int) -> list[float]:
    """First 25 days: moderate range. Last 5 days: push toward 20d high."""
    closes: list[float] = []
    for i in range(days):
        if i < days - 5:
            chg = rng.normal(0, 0.015)
            prev = closes[-1] if closes else base
            closes.append(max(prev * (1 + chg), 0.001))
        else:
            # Push up toward the 20d high
            twenty_high = max(closes) if closes else base * 1.05
            prev = closes[-1]
            closes.append(min(prev * (1 + rng.uniform(0.01, 0.025)), twenty_high * 0.98))
    return closes


def _gen_moderate_closes(rng: np.random.Generator, base: float, days: int) -> list[float]:
    closes = [base]
    for _ in range(days - 1):
        chg = rng.normal(0, 0.018)
        closes.append(max(closes[-1] * (1 + chg), 0.001))
    return closes


def _gen_erratic_closes(rng: np.random.Generator, base: float, days: int) -> list[float]:
    closes = [base]
    for _ in range(days - 1):
        chg = rng.normal(0, 0.035)
        closes.append(max(closes[-1] * (1 + chg), 0.001))
    return closes


def _gen_strong_accum_volume(rng: np.random.Generator, base_vol: int, days: int) -> list[float]:
    """
    Very quiet for first 15 days (0.3-0.7x), then aggressive spike last 5 days (4-8x).
    Produces volume_ratio_5d > 3.0 reliably.
    """
    vols = []
    for i in range(days):
        if i >= days - 5:
            vols.append(base_vol * rng.uniform(4.0, 8.0))
        else:
            vols.append(base_vol * rng.uniform(0.3, 0.7))
    return vols


def _gen_trending_volume(rng: np.random.Generator, base_vol: int, days: int) -> list[float]:
    """Volume trending upward; last 5 days run 2.5-4x the average."""
    vols = []
    for i in range(days):
        if i >= days - 5:
            multiplier = rng.uniform(2.5, 4.0)
        else:
            multiplier = rng.uniform(0.6, 1.2)
        vols.append(base_vol * multiplier)
    return vols


def _gen_spike_volume(rng: np.random.Generator, base_vol: int, days: int) -> list[float]:
    """Flat volume, then a 3x spike in the last 5 days."""
    vols = []
    for i in range(days):
        if i >= days - 5:
            vols.append(base_vol * rng.uniform(2.5, 4.0))
        else:
            vols.append(base_vol * rng.uniform(0.7, 1.3))
    return vols


def _gen_flat_volume(rng: np.random.Generator, base_vol: int, days: int, noise: float = 0.3) -> list[float]:
    return [base_vol * rng.uniform(1 - noise, 1 + noise) for _ in range(days)]


# ---------------------------------------------------------------------------
# Insider trade generators
# ---------------------------------------------------------------------------

def _make_trade(
    ticker: str,
    insider: dict,
    transaction_type: str,
    trade_date: pd.Timestamp,
    filing_lag_days: int,
    price: float,
    quantity: int,
    nb_owned_before: int,
    general_remarks: str = "",
) -> dict:
    value = round(price * quantity, 2)
    nb_owned_after = nb_owned_before + (quantity if transaction_type == "Buy" else -quantity)
    delta_own = round((quantity / nb_owned_before * 100) if nb_owned_before > 0 else 0.0, 2)
    if transaction_type != "Buy":
        delta_own = -delta_own
    return {
        "filing_date":    trade_date + timedelta(days=filing_lag_days),
        "trade_date":     trade_date,
        "ticker":         ticker,
        "issuer":         _ISSUERS.get(ticker, ticker),
        "insider_name":   insider["name"],
        "role":           insider["role"],
        "transaction_type": transaction_type,
        "price_cad":      price,
        "quantity":       quantity,
        "value_cad":      value,
        "nb_owned_after": nb_owned_after,
        "delta_own_pct":  delta_own,
        "general_remarks": general_remarks,
        "market_cap_cad": _MARKET_CAPS.get(ticker, 5_000_000),
    }


def _add_strong_accum_trades(
    rows: list, ticker: str, insiders: list, base_price: float,
    ref: pd.Timestamp,
) -> None:
    """3-5 buys in last 7 days by 2-3 distinct insiders, high value."""
    schedules = [
        (insiders[0], ref - timedelta(days=2), 500_000, 2),
        (insiders[min(1, len(insiders)-1)], ref - timedelta(days=4), 280_000, 3),
        (insiders[min(2, len(insiders)-1)], ref - timedelta(days=5), 150_000, 3),
    ]
    if len(insiders) > 1 and ticker == "VLE.V":
        schedules.append((insiders[0], ref - timedelta(days=1), 200_000, 1))

    nb_owned = 1_500_000
    for insider, trade_date, value_target, lag in schedules:
        qty = max(int(value_target / base_price), 1)
        rows.append(_make_trade(
            ticker, insider, "Buy", trade_date, lag,
            base_price, qty, nb_owned,
            "Acquisition in the public markets",
        ))
        nb_owned += qty


def _add_breakout_trades(
    rows: list, ticker: str, insiders: list, base_price: float,
    ref: pd.Timestamp,
) -> None:
    """1-2 buys 10-20 days ago — older signal, volume will do the talking."""
    schedules = [
        (insiders[0], ref - timedelta(days=14), 60_000, 4),
    ]
    if len(insiders) > 1:
        schedules.append((insiders[1], ref - timedelta(days=18), 35_000, 5))

    nb_owned = 800_000
    for insider, trade_date, value_target, lag in schedules:
        qty = max(int(value_target / base_price), 1)
        rows.append(_make_trade(
            ticker, insider, "Buy", trade_date, lag,
            base_price, qty, nb_owned,
            "Open market purchase",
        ))
        nb_owned += qty


def _add_mild_accum_trades(
    rows: list, ticker: str, insiders: list, base_price: float,
    ref: pd.Timestamp,
) -> None:
    """1-2 buys 8-14 days ago by a single insider."""
    trade_date = ref - timedelta(days=random.randint(8, 14))
    value_target = random.randint(20_000, 55_000)
    qty = max(int(value_target / base_price), 1)
    rows.append(_make_trade(
        ticker, insiders[0], "Buy", trade_date, 3,
        base_price, qty, 500_000,
        "Open market purchase",
    ))


def _add_no_signal_trades(
    rows: list, ticker: str, insiders: list, base_price: float,
    ref: pd.Timestamp,
) -> None:
    """Sells or option exercises only — never a Buy."""
    trade_date = ref - timedelta(days=random.randint(5, 25))
    qty = random.randint(50_000, 200_000)
    tx_type = random.choice(["Sell", "Option Exercise"])
    remarks = "Disposition in the public markets" if tx_type == "Sell" else "Option exercise"
    rows.append(_make_trade(
        ticker, insiders[0], tx_type, trade_date, 4,
        base_price, qty, 2_000_000, remarks,
    ))
