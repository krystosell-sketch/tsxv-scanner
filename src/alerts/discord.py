"""
Discord webhook alerts for the top daily TSXV/CSE setups.

Sends up to 10 embeds per message (Discord limit).  Each embed contains:
  - Ticker, grade, and composite score in the title
  - Signals (ACCUM / PRE-BRK / CLST) as a field
  - Latest insider name, role, and dollar amount
  - Claude-generated explanation in the description (optional)

Usage:
    from src.alerts.discord import send_daily_alert

    success = send_daily_alert(detected_df, explanations=explanations, top_n=5)
"""

import logging
from datetime import date
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grade → Discord embed colour (0xRRGGBB)
# ---------------------------------------------------------------------------
_GRADE_COLOUR: dict[str, int] = {
    "A+": 0x00C853,   # bright green
    "A":  0x76FF03,   # light green
    "B":  0xFFD600,   # amber
    "C":  0x9E9E9E,   # grey
}

_GRADE_EMOJI: dict[str, str] = {
    "A+": "🏆",
    "A":  "⭐",
    "B":  "👍",
    "C":  "👀",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_value(cad: float) -> str:
    if cad >= 1_000_000:
        return f"${cad / 1_000_000:.1f}M"
    if cad >= 1_000:
        return f"${cad / 1_000:.0f}k"
    return f"${cad:.0f}"


def _fmt_flags(row: dict[str, Any]) -> str:
    parts = []
    if row.get("accumulation_flag"):
        parts.append("🔵 ACCUM")
    if row.get("pre_breakout_flag"):
        parts.append("🟡 PRE-BRK")
    if row.get("cluster_buying_flag"):
        parts.append("🟣 CLST")
    return " | ".join(parts) if parts else "—"


def _fmt_insider(row: dict[str, Any]) -> str:
    name  = str(row.get("insider_name") or "").strip()
    role  = str(row.get("latest_role") or "").strip()
    value = float(row.get("latest_value_cad") or 0)
    ts    = str(row.get("latest_trade_date") or "").strip()

    if not name or name in ("nan", "None", ""):
        return "N/A"

    parts = [f"{name} ({role})" if role else name, _fmt_value(value)]
    if ts and ts not in ("", "nan", "NaT", "None"):
        try:
            import pandas as pd
            parts.append(pd.Timestamp(ts).strftime("%Y-%m-%d"))
        except Exception:  # noqa: BLE001
            pass
    return " · ".join(parts)


def _ticker_links(ticker: str, issuer: str = "") -> str:
    """Construit les liens directs TSXTracker selon l'exchange."""
    clean = ticker.replace(".CN", "").replace(".V", "")

    if ticker.endswith(".CN"):
        exchange = "CSE"
    else:
        exchange = "TSXV"

    tsx_url = f"https://tsxtracker.com/en/{exchange}/{clean}"
    return f"[TSXTracker — {ticker}]({tsx_url})"


def _build_embed(rank: int, row: dict[str, Any], explanation: str) -> dict:
    """Build a single Discord embed dict for one setup."""
    ticker = str(row.get("ticker", "N/A"))
    grade  = str(row.get("grade", "?"))
    score  = float(row.get("composite_score") or 0)
    emoji  = _GRADE_EMOJI.get(grade, "")
    colour = _GRADE_COLOUR.get(grade, 0x9E9E9E)

    description = explanation.strip() if explanation else "_No AI explanation available._"

    issuer = str(row.get("issuer") or "").strip()
    issuer_line = f" · *{issuer}*" if issuer and issuer not in ("nan", "None", "") else ""

    # Signal reasons (bullet points)
    reasons = row.get("signal_reasons", [])
    if isinstance(reasons, str):
        reasons = [r for r in reasons.split("|") if r.strip()]
    reasons_str = "\n".join(f"• {r}" for r in reasons) if reasons else "—"

    # AI explanation (if available) + signal reasons
    signals_block = f"**Signaux détectés**\n{reasons_str}"
    if description and description != "_No AI explanation available._":
        full_description = f"{description}\n\n{signals_block}"
    else:
        full_description = signals_block

    embed: dict[str, Any] = {
        "title": f"#{rank} · {ticker}{issuer_line}   {emoji} Grade {grade}  |  Score {score:.1f}",
        "description": full_description,
        "color": colour,
        "fields": [
            {
                "name": "Signals",
                "value": _fmt_flags(row),
                "inline": True,
            },
            {
                "name": "Latest Insider",
                "value": _fmt_insider(row),
                "inline": False,
            },
            {
                "name": "Vérifier",
                "value": _ticker_links(ticker, issuer),
                "inline": False,
            },
        ],
        "footer": {
            "text": f"TSXV/CSE Insider Scanner · {date.today().strftime('%Y-%m-%d')}"
        },
    }
    return embed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_daily_alert(
    detected_df,
    explanations: dict[str, str] | None = None,
    top_n: int = 5,
    webhook_url: str = "",
) -> bool:
    """
    Send the top N setups as a Discord webhook message.

    Args:
        detected_df  : Full pipeline output DataFrame (sorted by score desc).
        explanations : Dict {ticker: explanation} from explain_top_setups().
                       Pass None or {} to skip AI descriptions.
        top_n        : Number of setups to include (Discord cap: 10 embeds).
        webhook_url  : Override webhook URL.  Falls back to DISCORD_WEBHOOK_URL
                       from config (which reads DISCORD_WEBHOOK_URL env var).

    Returns:
        True if the request succeeded (HTTP 2xx), False otherwise.
    """
    from src.config import DISCORD_WEBHOOK_URL as _cfg_url

    url = webhook_url or _cfg_url
    if not url:
        logger.warning("No DISCORD_WEBHOOK_URL set — skipping Discord alert")
        return False

    if detected_df is None or detected_df.empty:
        logger.info("No setups to send to Discord")
        return False

    explanations = explanations or {}
    top_n = min(top_n, 10)  # Discord hard cap
    top   = detected_df.head(top_n)

    embeds = [
        _build_embed(rank, row.to_dict(), explanations.get(str(row["ticker"]), ""))
        for rank, (_, row) in enumerate(top.iterrows(), start=1)
    ]

    n_active = int((detected_df.get("insider_buy_30d", 0) > 0).sum()) if "insider_buy_30d" in detected_df.columns else "?"
    header = (
        f"📊 **Daily Insider Accumulation Scan — {date.today().strftime('%Y-%m-%d')}**\n"
        f"Universe: {len(detected_df)} tickers · Showing top {len(embeds)}"
    )

    payload: dict[str, Any] = {
        "username": "TSXV/CSE Insider Scanner",
        "content": header,
        "embeds": embeds,
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        logger.info("Discord alert sent (HTTP %d)", resp.status_code)
        return True
    except requests.exceptions.HTTPError as exc:
        logger.error(
            "Discord webhook HTTP error: %s — body: %s",
            exc,
            getattr(exc.response, "text", "")[:300],
        )
        return False
    except requests.exceptions.RequestException as exc:
        logger.error("Discord webhook request failed: %s", exc)
        return False
