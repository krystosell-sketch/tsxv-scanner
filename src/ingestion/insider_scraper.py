"""
Scrapes insider trades from the TSXTracker public API.

API: https://api.tsxtracker.com/v0/transactions
- Returns 100 rows per page
- Each page includes enriched lookup tables: insiders, issuers, tickers
- Exchange IDs: TSXV=3, CSE=5
- trnNatureCode=10: open-market acquisition/disposition (nb>0 = buy)
- titles bitmask: 16=SeniorOfficer, 8=Director, 4=TenPercentHolder
"""

import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_API_BASE     = "https://api.tsxtracker.com/v0"
_ENDPOINT     = f"{_API_BASE}/transactions"
_PAGE_SIZE    = 100
_EXCHANGE_IDS = {3, 5}          # TSXV=3, CSE=5
_BUY_CODES    = {10, 11}        # open-market & private acquisitions/dispositions

_EXCHANGE_SUFFIX = {3: ".V", 5: ".CN"}

# titles bitmask → canonical role
_TITLES_ROLES = [
    (16, "CEO"),          # SeniorOfficer → maps to CEO bucket (weight 3)
    (8,  "Director"),     # Director
    (4,  "10%Holder"),    # TenPercentHolder
]

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":     "application/json",
    "Referer":    "https://tsxtracker.com/",
    "Origin":     "https://tsxtracker.com",
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_insider_trades(
    lookback_days: int = 45,
    max_pages: int = 12,
    request_delay: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch recent insider trades from TSXTracker API for TSXV and CSE stocks.

    Returns a DataFrame with the same schema expected by normalize_insider_df():
        filing_date, trade_date, ticker, issuer, insider_name, role,
        transaction_type, price_cad, quantity, value_cad, nb_owned_after,
        delta_own_pct, general_remarks, market_cap_cad

    market_cap_cad is set to 0 (unknown from this source — enriched separately).
    """
    begin_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    rows: list[dict] = []

    for page in range(1, max_pages + 1):
        params = {
            "page":           page,
            "limit":          _PAGE_SIZE,
            "trnNatureCodes": ",".join(str(c) for c in _BUY_CODES),
            "beginDate":      begin_date,
            "orderBy":        "filingDate",
            "order":          "desc",
        }

        try:
            resp = requests.get(_ENDPOINT, params=params, headers=_HEADERS, timeout=15)
            if resp.status_code == 429:
                logger.warning("Rate limited on page %d — waiting 10s before retry", page)
                time.sleep(10)
                resp = requests.get(_ENDPOINT, params=params, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            logger.error("API request failed (page %d): %s", page, exc)
            break

        transactions  = payload.get("transactions", [])
        insider_lut   = {i["id"]: i for i in payload.get("insiders", [])}
        issuer_lut    = {i["id"]: i for i in payload.get("issuers", [])}
        # Keep only valid TSXV/CSE tickers (one ticker per issuerId, prefer non-invalid)
        ticker_lut    = _build_ticker_lut(payload.get("tickers", []))

        logger.debug("Page %d: %d transactions, %d TSXV/CSE tickers",
                     page, len(transactions), len(ticker_lut))

        if not transactions:
            logger.info("No more transactions at page %d", page)
            break

        for trn in transactions:
            row = _parse_transaction(trn, insider_lut, issuer_lut, ticker_lut)
            if row:
                rows.append(row)

        total = payload.get("total", 0)
        fetched_so_far = page * _PAGE_SIZE
        if fetched_so_far >= total:
            logger.info("Fetched all %d transactions across %d pages", total, page)
            break

        if request_delay > 0:
            time.sleep(request_delay)

    df = pd.DataFrame(rows) if rows else _empty_df()
    logger.info("fetch_insider_trades: %d rows (lookback %d days)", len(df), lookback_days)
    return df


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _build_ticker_lut(tickers: list[dict]) -> dict[int, dict]:
    """
    Returns {issuerId: ticker_dict} keeping only valid TSXV/CSE tickers.
    Prefers non-invalid entries; among multiple exchanges picks TSXV over CSE.
    """
    candidates: dict[int, dict] = {}
    for t in tickers:
        if t.get("exchangeId") not in _EXCHANGE_IDS:
            continue
        if t.get("isInvalid"):
            continue
        issuer_id = t["issuerId"]
        existing = candidates.get(issuer_id)
        if existing is None:
            candidates[issuer_id] = t
        else:
            # Prefer TSXV (3) over CSE (5)
            if t.get("exchangeId") == 3 and existing.get("exchangeId") != 3:
                candidates[issuer_id] = t
    return candidates


def _parse_transaction(
    trn: dict,
    insider_lut: dict,
    issuer_lut: dict,
    ticker_lut: dict,
) -> dict | None:
    """
    Parse a single transaction dict into our insider_df row schema.
    Returns None if not a TSXV/CSE buy, or if critical data is missing.
    """
    issuer_id  = trn.get("issuerId")
    insider_id = trn.get("insiderId")
    nb         = trn.get("nb")

    # Must be a purchase (nb > 0) and belong to a TSXV/CSE issuer
    if nb is None or nb <= 0:
        return None
    if issuer_id not in ticker_lut:
        return None

    ticker_rec = ticker_lut[issuer_id]
    insider    = insider_lut.get(insider_id, {})
    issuer     = issuer_lut.get(issuer_id, {})

    exchange_id  = ticker_rec.get("exchangeId", 0)
    suffix       = _EXCHANGE_SUFFIX.get(exchange_id, "")
    ticker_sym   = f"{ticker_rec.get('name', '')}{suffix}"

    price  = trn.get("price")
    value  = trn.get("value")
    if value is None and price is not None:
        value = price * nb

    closing_bal  = trn.get("closingBalance") or 0
    bal_pct      = trn.get("balanceChangePct") or 0.0
    role         = _decode_titles(insider.get("titles", 0))

    return {
        "filing_date":       _parse_iso(trn.get("filingDate")),
        "trade_date":        _parse_iso(trn.get("trnDate")),
        "ticker":            ticker_sym,
        "issuer":            issuer.get("name", ""),
        "insider_name":      insider.get("name", ""),
        "role":              role,
        "transaction_type":  "Buy",
        "price_cad":         float(price) if price is not None else 0.0,
        "quantity":          int(nb),
        "value_cad":         float(value) if value is not None else 0.0,
        "nb_owned_after":    int(closing_bal),
        "delta_own_pct":     float(bal_pct),
        "general_remarks":   trn.get("GeneralRemarks") or "Open market purchase",
        "market_cap_cad":    0.0,  # enriched separately
    }


def _decode_titles(titles: int) -> str:
    """Map SEDI titles bitmask to canonical role string."""
    for bit, role in _TITLES_ROLES:
        if titles & bit:
            return role
    return "Other"


def _parse_iso(dt_str: str | None) -> pd.Timestamp | None:
    if not dt_str:
        return None
    try:
        return pd.Timestamp(dt_str).tz_localize(None)
    except Exception:
        return None


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "filing_date", "trade_date", "ticker", "issuer", "insider_name",
        "role", "transaction_type", "price_cad", "quantity", "value_cad",
        "nb_owned_after", "delta_own_pct", "general_remarks", "market_cap_cad",
    ])
