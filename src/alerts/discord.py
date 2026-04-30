"""
Alertes Discord pour les top setups TSXV/CSE.

Envoie :
  - Un embed de résumé global (stats du scan)
  - Un embed par setup (top N) avec signaux, insider, lien TSXTracker
  - Un graphique matplotlib en pièce jointe
"""

import json
import logging
from datetime import date
from typing import Any

import requests

logger = logging.getLogger(__name__)

_GRADE_COLOUR = {"A+": 0x00C853, "A": 0x76FF03, "B": 0xFFD600, "C": 0x9E9E9E}
_GRADE_EMOJI  = {"A+": "🏆", "A": "⭐", "B": "👍", "C": "👀"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_value(cad: float) -> str:
    if cad >= 1_000_000:
        return f"${cad / 1_000_000:.1f}M"
    if cad >= 1_000:
        return f"${cad / 1_000:.0f}k"
    return f"${cad:.0f}"


def _fmt_flags(row: dict) -> str:
    parts = []
    if row.get("accumulation_flag"):   parts.append("🔵 ACCUM")
    if row.get("pre_breakout_flag"):   parts.append("🟡 PRE-BRK")
    if row.get("cluster_buying_flag"): parts.append("🟣 CLST")
    return " | ".join(parts) if parts else "—"


def _fmt_insider(row: dict) -> str:
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
        except Exception:
            pass
    return " · ".join(parts)


def _ticker_links(ticker: str) -> str:
    clean    = ticker.replace(".CN", "").replace(".V", "")
    exchange = "CSE" if ticker.endswith(".CN") else "TSXV"
    url      = f"https://tsxtracker.com/en/{exchange}/{clean}"
    return f"[TSXTracker — {ticker}]({url})"


def _signal_reasons(row: dict) -> str:
    reasons = row.get("signal_reasons", [])
    if isinstance(reasons, str):
        reasons = [r for r in reasons.split("|") if r.strip()]
    return "\n".join(f"• {r}" for r in reasons) if reasons else "—"


# ---------------------------------------------------------------------------
# Embeds
# ---------------------------------------------------------------------------

def _summary_embed(detected_df) -> dict:
    """Embed de résumé global du scan."""
    n_total  = len(detected_df)
    n_active = int((detected_df.get("insider_buy_30d", 0) > 0).sum()) \
               if "insider_buy_30d" in detected_df.columns else "?"
    n_accum  = int(detected_df["accumulation_flag"].sum())
    n_pre    = int(detected_df["pre_breakout_flag"].sum())
    n_clst   = int(detected_df["cluster_buying_flag"].sum())

    grade_counts = detected_df["grade"].value_counts()
    grades_str = "  ".join(
        f"{g}: **{grade_counts.get(g, 0)}**"
        for g in ["A+", "A", "B", "C"]
    )

    return {
        "title": f"📊 Scan quotidien — {date.today().strftime('%Y-%m-%d')}",
        "description": (
            f"**{n_total}** tickers scannés  |  "
            f"**{n_active}** avec activité insider (30j)\n\n"
            f"**Signaux**  🔵 ACCUM: {n_accum}  🟡 PRE-BRK: {n_pre}  🟣 CLST: {n_clst}\n"
            f"**Grades**  {grades_str}"
        ),
        "color": 0x5865F2,
    }


def _setup_embed(rank: int, row: dict, explanation: str) -> dict:
    """Embed pour un setup individuel."""
    ticker  = str(row.get("ticker", "N/A"))
    grade   = str(row.get("grade", "?"))
    score   = float(row.get("composite_score") or 0)
    emoji   = _GRADE_EMOJI.get(grade, "")
    colour  = _GRADE_COLOUR.get(grade, 0x9E9E9E)
    issuer  = str(row.get("issuer") or "").strip()

    issuer_line = f" · *{issuer}*" if issuer and issuer not in ("nan", "None", "") else ""

    # Description : explication Claude + signaux détectés
    signals_block = f"**Signaux détectés**\n{_signal_reasons(row)}"
    if explanation and explanation not in ("", "_No AI explanation available._"):
        description = f"{explanation}\n\n{signals_block}"
    else:
        description = signals_block

    return {
        "title": f"#{rank} · {ticker}{issuer_line}  {emoji} Grade {grade}  |  Score {score:.1f}",
        "description": description,
        "color": colour,
        "fields": [
            {
                "name": "Dernier achat insider",
                "value": _fmt_insider(row),
                "inline": False,
            },
            {
                "name": "Drapeaux",
                "value": _fmt_flags(row),
                "inline": True,
            },
            {
                "name": "Liens",
                "value": _ticker_links(ticker),
                "inline": True,
            },
        ],
        "footer": {
            "text": f"TSXV/CSE Insider Scanner · {date.today().strftime('%Y-%m-%d')}"
        },
    }


# ---------------------------------------------------------------------------
# Envoi
# ---------------------------------------------------------------------------

def send_daily_alert(
    detected_df,
    explanations: dict[str, str] | None = None,
    top_n: int = 10,
    webhook_url: str = "",
) -> bool:
    """
    Envoie l'alerte Discord : résumé + top N setups + graphique PNG.

    Args:
        detected_df  : DataFrame pipeline trié par score desc.
        explanations : Dict {ticker: explication Claude}.
        top_n        : Nombre de setups (max 9 embeds + 1 résumé = 10 total).
        webhook_url  : Override URL webhook.

    Returns:
        True si succès, False sinon.
    """
    from src.config import DISCORD_WEBHOOK_URL as _cfg_url
    url = webhook_url or _cfg_url

    if not url:
        logger.warning("Pas de DISCORD_WEBHOOK_URL — alerte Discord ignorée")
        return False

    if detected_df is None or detected_df.empty:
        logger.info("Aucun setup à envoyer sur Discord")
        return False

    explanations = explanations or {}
    top_n  = min(top_n, 9)   # 1 résumé + 9 setups = 10 max (limite Discord)
    top    = detected_df.head(top_n)

    embeds = [_summary_embed(detected_df)]
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        ticker      = str(row["ticker"])
        explanation = explanations.get(ticker, "")
        embeds.append(_setup_embed(rank, row.to_dict(), explanation))

    # Générer le graphique
    chart_bytes = None
    try:
        from src.alerts.chart import generate_score_chart
        chart_bytes = generate_score_chart(detected_df, top_n=top_n)
    except Exception as exc:
        logger.warning("Impossible de générer le graphique: %s", exc)

    # Envoyer
    try:
        if chart_bytes:
            # Multipart : payload JSON + image PNG
            payload = {
                "username": "TSXV/CSE Insider Scanner",
                "embeds": embeds,
            }
            files = {
                "payload_json": (None, json.dumps(payload), "application/json"),
                "files[0]": ("scores.png", chart_bytes, "image/png"),
            }
            resp = requests.post(url, files=files, timeout=20)
        else:
            # Fallback sans image
            payload = {
                "username": "TSXV/CSE Insider Scanner",
                "embeds": embeds,
            }
            resp = requests.post(url, json=payload, timeout=15)

        resp.raise_for_status()
        logger.info("Alerte Discord envoyée (HTTP %d)", resp.status_code)
        return True

    except requests.exceptions.HTTPError as exc:
        logger.error("Erreur HTTP Discord: %s — %s",
                     exc, getattr(exc.response, "text", "")[:300])
        return False
    except requests.exceptions.RequestException as exc:
        logger.error("Erreur requête Discord: %s", exc)
        return False
