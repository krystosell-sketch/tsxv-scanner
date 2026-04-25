"""
Fetches OHLCV price/volume data via yfinance.

Ticker format:
  TSXV: TICKER.V   (e.g. ABC.V)
  CSE:  TICKER.CN  (e.g. XYZ.CN)

yfinance is rate-limited; we batch with a short delay between tickers.
"""

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_DEFAULT_DAYS   = 30
_REQUEST_DELAY  = 0.3   # seconds between individual ticker fetches
_BATCH_SIZE     = 20    # tickers per yfinance batch download


def fetch_ohlcv(
    tickers: list[str],
    days: int = _DEFAULT_DAYS,
    batch_size: int = _BATCH_SIZE,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for a list of tickers (with .V or .CN suffix).

    Returns ohlcv_df with columns:
        date, ticker, open, high, low, close, volume

    Tickers that yfinance cannot resolve are silently skipped.
    """
    if not tickers:
        return _empty_df()

    end_date   = datetime.now()
    start_date = end_date - timedelta(days=days + 5)  # +5 to account for holidays

    all_rows: list[dict] = []
    unique_tickers = list(dict.fromkeys(tickers))  # deduplicate, preserve order

    # Process in batches to respect rate limits
    for batch_start in range(0, len(unique_tickers), batch_size):
        batch = unique_tickers[batch_start:batch_start + batch_size]
        rows = _fetch_batch(batch, start_date, end_date)
        all_rows.extend(rows)
        if batch_start + batch_size < len(unique_tickers):
            time.sleep(_REQUEST_DELAY)

    df = pd.DataFrame(all_rows) if all_rows else _empty_df()
    logger.info("fetch_ohlcv: %d rows for %d tickers (%d days)",
                len(df), df["ticker"].nunique() if not df.empty else 0, days)
    return df


def fetch_market_caps(tickers: list[str]) -> dict[str, float]:
    """
    Fetch market capitalisation (CAD) for each ticker.
    Falls back to 0 if unavailable.
    Returns {ticker: market_cap_cad}.
    """
    result: dict[str, float] = {}
    for tkr in tickers:
        try:
            info = yf.Ticker(tkr).info
            mktcap = float(info.get("marketCap") or 0)
            # yfinance returns market cap in the listed currency
            # For TSX/TSXV/CSE, this should already be CAD
            result[tkr] = mktcap
        except Exception as exc:
            logger.debug("market cap unavailable for %s: %s", tkr, exc)
            result[tkr] = 0.0
        time.sleep(_REQUEST_DELAY)
    logger.info("fetch_market_caps: %d tickers", len(result))
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_batch(
    tickers: list[str],
    start_date: datetime,
    end_date: datetime,
) -> list[dict]:
    """Download a batch of tickers with yfinance and return OHLCV rows."""
    rows: list[dict] = []

    if len(tickers) == 1:
        raw = _single_ticker_download(tickers[0], start_date, end_date)
        if raw is not None and not raw.empty:
            rows.extend(_flatten_single(raw, tickers[0]))
        return rows

    try:
        raw = yf.download(
            tickers,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )
    except Exception as exc:
        logger.warning("Batch download failed (%s), falling back to singles", exc)
        for tkr in tickers:
            single = _single_ticker_download(tkr, start_date, end_date)
            if single is not None and not single.empty:
                rows.extend(_flatten_single(single, tkr))
        return rows

    if raw.empty:
        return rows

    # Multi-ticker: yfinance returns MultiIndex (ticker, field)
    for tkr in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                # Level 0 = ticker name, level 1 = OHLCV field
                if tkr in raw.columns.get_level_values(0):
                    tkr_df = raw.xs(tkr, axis=1, level=0).dropna(how="all")
                else:
                    continue
            else:
                tkr_df = raw.dropna(how="all")
                if len(tickers) > 1:
                    continue  # can't separate in flat mode
            rows.extend(_flatten_single(tkr_df, tkr))
        except Exception as exc:
            logger.debug("Could not extract %s from batch: %s", tkr, exc)

    return rows


def _single_ticker_download(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame | None:
    """Download a single ticker quietly, return None on failure."""
    try:
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        return df if not df.empty else None
    except Exception as exc:
        logger.debug("Failed to download %s: %s", ticker, exc)
        return None


def _flatten_single(df: pd.DataFrame, ticker: str) -> list[dict]:
    """Convert a single-ticker OHLCV DataFrame into a list of row dicts."""
    rows: list[dict] = []
    # Normalise column names (yfinance may return MultiIndex or flat)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    df.columns = [c.lower() for c in df.columns]

    for date, row in df.iterrows():
        close = row.get("close") or row.get("adj close")
        if close is None or pd.isna(close) or float(close) <= 0:
            continue
        rows.append({
            "date":   pd.Timestamp(date).normalize(),
            "ticker": ticker,
            "open":   round(float(row.get("open", close)), 4),
            "high":   round(float(row.get("high", close)), 4),
            "low":    round(float(row.get("low",  close)), 4),
            "close":  round(float(close), 4),
            "volume": int(row.get("volume", 0) or 0),
        })
    return rows


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
