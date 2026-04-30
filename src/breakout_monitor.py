"""
Moniteur de breakout en temps réel — GitHub Actions (toutes les 15 min).

Logique :
  1. Charge data/latest.json (résultats du scan de la veille/du soir)
  2. Filtre les tickers en phase "Attente" : ACCUM sans PRE-BRK
  3. Récupère les prix frais yfinance pour ces tickers seulement
  4. Recalcule volume_spike + near_breakout avec les données fraîches
  5. Pour chaque nouveau PRE-BRK non encore alerté aujourd'hui :
       → Envoie alerte Discord courte 🚨
  6. Persiste la liste "alerté aujourd'hui" (cache GitHub Actions, reset quotidien)

Aucun scrape TSXTracker — les données insiders viennent de latest.json.
"""

import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# Assure que le package src est importable depuis la racine du projet
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

_LATEST_JSON    = Path("data/latest.json")
_ALERTED_TODAY  = Path("data/alerted_today.json")


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def _load_latest() -> list[dict]:
    """Charge data/latest.json. Retourne une liste vide si absent."""
    if not _LATEST_JSON.exists():
        logger.warning("data/latest.json introuvable — scan de fin de journée pas encore exécuté.")
        return []
    with open(_LATEST_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("tickers", [])


def _load_alerted_today() -> set[str]:
    """Charge la liste des tickers déjà alertés aujourd'hui."""
    if not _ALERTED_TODAY.exists():
        return set()
    with open(_ALERTED_TODAY, encoding="utf-8") as f:
        data = json.load(f)
    # Reset si le fichier date d'un autre jour
    if data.get("date") != date.today().isoformat():
        return set()
    return set(data.get("tickers", []))


def _save_alerted_today(alerted: set[str]) -> None:
    """Persiste la liste des tickers alertés aujourd'hui."""
    _ALERTED_TODAY.parent.mkdir(parents=True, exist_ok=True)
    with open(_ALERTED_TODAY, "w", encoding="utf-8") as f:
        json.dump({"date": date.today().isoformat(), "tickers": list(alerted)}, f)


# ---------------------------------------------------------------------------
# Détection de breakout avec prix frais
# ---------------------------------------------------------------------------

def _check_breakouts(attente_rows: list[dict]) -> list[dict]:
    """
    Pour chaque ticker en Attente, vérifie si les conditions PRE-BRK
    sont maintenant réunies avec les prix frais yfinance.

    PRE-BRK = volume_spike (ratio >= 2.0×) ET near_breakout (close >= 90% du range 20j)

    Returns:
        Liste de dicts pour les tickers où le breakout vient de se déclencher.
    """
    from src.config import NEAR_BREAKOUT_THRESHOLD, VOLUME_SPIKE_MULTIPLIER
    from src.ingestion.market_data import fetch_ohlcv
    import numpy as np

    tickers = [r["ticker"] for r in attente_rows]
    if not tickers:
        return []

    logger.info("Fetch prix frais pour %d tickers Attente...", len(tickers))
    ohlcv_df = fetch_ohlcv(tickers, days=25)

    if ohlcv_df.empty:
        logger.warning("Aucun prix récupéré depuis yfinance.")
        return []

    breakouts = []

    for row in attente_rows:
        ticker = row["ticker"]
        grp = ohlcv_df[ohlcv_df["ticker"] == ticker].sort_values("date")
        if len(grp) < 5:
            continue

        closes  = grp["close"].values.astype(float)
        volumes = grp["volume"].values.astype(float)

        # Volume spike : moyenne 5j vs moyenne 20j
        avg_20d = float(np.mean(volumes)) if len(volumes) > 0 else 1.0
        avg_5d  = float(np.mean(volumes[-5:])) if len(volumes) >= 5 else avg_20d
        vol_ratio    = avg_5d / avg_20d if avg_20d > 0 else 1.0
        volume_spike = vol_ratio >= VOLUME_SPIKE_MULTIPLIER

        # Near breakout : close >= close_low_20d + 90% du range
        close_high = float(np.max(closes[-20:])) if len(closes) >= 1 else closes[-1]
        close_low  = float(np.min(closes[-20:])) if len(closes) >= 1 else closes[-1]
        close_range = close_high - close_low
        last_close  = closes[-1]
        if close_range > 0:
            near_breakout = last_close >= (close_low + NEAR_BREAKOUT_THRESHOLD * close_range)
        else:
            near_breakout = False

        logger.debug(
            "%s — vol_ratio=%.2f spike=%s near_brk=%s",
            ticker, vol_ratio, volume_spike, near_breakout,
        )

        if volume_spike and near_breakout:
            # Enrichir le row avec les données fraîches pour l'embed Discord
            enriched = dict(row)
            enriched["volume_ratio_5d"] = round(vol_ratio, 2)
            breakouts.append(enriched)

    return breakouts


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_monitor() -> int:
    """
    Exécute le moniteur de breakout. Retourne le nombre d'alertes envoyées.
    """
    from src.alerts.discord import send_breakout_alert

    # 1. Charger les tickers en Attente
    all_tickers = _load_latest()
    attente = [
        r for r in all_tickers
        if r.get("accumulation_flag") and not r.get("pre_breakout_flag")
    ]

    if not attente:
        logger.info("Aucun ticker en Attente — moniteur terminé.")
        return 0

    logger.info("%d tickers en Attente à surveiller.", len(attente))

    # 2. Charger les tickers déjà alertés aujourd'hui
    alerted = _load_alerted_today()

    # 3. Détecter les nouveaux breakouts
    breakouts = _check_breakouts(attente)

    # 4. Alerter pour les nouveaux breakouts seulement
    sent = 0
    for row in breakouts:
        ticker = row["ticker"]
        if ticker in alerted:
            logger.info("%s — déjà alerté aujourd'hui, skip.", ticker)
            continue

        logger.info("🚨 Breakout détecté : %s", ticker)
        success = send_breakout_alert(row)
        if success:
            alerted.add(ticker)
            sent += 1

    # 5. Sauvegarder la liste mise à jour
    _save_alerted_today(alerted)

    logger.info("Moniteur terminé — %d alerte(s) envoyée(s).", sent)
    return sent


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    # Charger .env si disponible
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    n = run_monitor()
    print(f"[OK] Moniteur de breakout termine — {n} alerte(s) envoyee(s).")


if __name__ == "__main__":
    main()
