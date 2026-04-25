"""
Mini-backtest engine et rapport de performance des signaux.

Flux de données :
  scored_setups (DB)
       │
       ▼
  load_all_scored_setups()          ← tous les signaux historiques
       │
       ▼
  fetch_ohlcv(tickers, 90j)         ← prix frais yfinance
       │
       ▼
  upsert_daily_prices()             ← étend daily_prices
       │
       ▼
  load_all_daily_prices()           ← historique complet (sans filtre date)
       │
       ▼
  _compute_returns(signals, prices) ← jointure + lookup T+5/10/20
       │
       ▼
  upsert_backtest_results()         ← persistance
       │
       ▼
  generate_performance_report()     ← rapport console

Fonctions pures (sans I/O) : _get_close_after, _compute_returns,
    generate_performance_report et ses helpers internes.
Effets de bord isolés dans : run_backtest.
"""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "scanner.db"


# ---------------------------------------------------------------------------
# Helpers purs
# ---------------------------------------------------------------------------

def _get_close_after(
    prices_df: pd.DataFrame,
    ticker: str,
    ref_date: pd.Timestamp,
    offset_days: int,
) -> float | None:
    """
    Retourne le close du premier jour de trading >= ref_date + offset_days.

    Sémantique "nearest after or equal" : si la date cible tombe un weekend
    ou un jour férié, utilise le prochain jour de trading disponible.

    Args:
        prices_df : DataFrame avec colonnes [date (datetime64), ticker, close],
                    trié par (ticker, date) croissant.
        ticker    : Symbole boursier à rechercher (ex. 'VLE.V').
        ref_date  : Date du signal (T+0), pd.Timestamp.
        offset_days: Décalage calendaire à ajouter (0, 5, 10 ou 20).

    Returns:
        float close si trouvé, None si ticker absent ou pas de données futures.
    """
    target = ref_date + pd.Timedelta(days=offset_days)
    sub = prices_df[
        (prices_df["ticker"] == ticker) & (prices_df["date"] >= target)
    ]
    if sub.empty:
        return None
    return float(sub.iloc[0]["close"])


def _compute_returns(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    offsets: list[int],
) -> pd.DataFrame:
    """
    Joint signals × prices et calcule les retours % à chaque offset.

    Pour chaque (run_date, ticker) dans signals_df :
      - close_at_signal = close sur run_date (ou prochain jour de trading)
      - close_tN        = close sur premier jour >= run_date + N jours
      - return_Nd       = (close_tN - close_at_signal) / close_at_signal * 100

    Les signaux sans prix (ticker introuvable, données futures manquantes)
    sont conservés avec des colonnes return = None pour la traçabilité.

    Args:
        signals_df : Sortie de load_all_scored_setups().
                     Colonnes requises : run_date, ticker, grade,
                     composite_score, accumulation_flag, pre_breakout_flag,
                     cluster_buying_flag.
        prices_df  : Sortie de load_all_daily_prices().
                     Colonnes requises : date (datetime64), ticker, close.
                     Trié par (ticker, date).
        offsets    : Liste de décalages calendaires, ex. [5, 10, 20].

    Returns:
        DataFrame avec le schéma backtest_results (une ligne par signal).
        Retours None pour les horizons sans données disponibles.
    """
    # Pré-trier une seule fois pour les slices internes
    prices_sorted = prices_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    results = []
    for _, sig in signals_df.iterrows():
        signal_date = pd.Timestamp(sig["run_date"])
        ticker = str(sig["ticker"])

        close_at_signal = _get_close_after(prices_sorted, ticker, signal_date, 0)

        row: dict = {
            "signal_date":         sig["run_date"],
            "ticker":              ticker,
            "grade":               sig.get("grade"),
            "composite_score":     sig.get("composite_score"),
            "accumulation_flag":   int(sig.get("accumulation_flag", 0)),
            "pre_breakout_flag":   int(sig.get("pre_breakout_flag", 0)),
            "cluster_buying_flag": int(sig.get("cluster_buying_flag", 0)),
            "close_at_signal":     close_at_signal,
        }

        for offset in offsets:
            close_key  = f"close_t{offset}"
            return_key = f"return_{offset}d"

            close_tn = _get_close_after(prices_sorted, ticker, signal_date, offset)
            row[close_key] = close_tn

            if close_at_signal and close_tn:
                row[return_key] = round(
                    (close_tn - close_at_signal) / close_at_signal * 100, 2
                )
            else:
                row[return_key] = None

        results.append(row)

    return pd.DataFrame(results) if results else pd.DataFrame()


# ---------------------------------------------------------------------------
# Orchestrateur (effets de bord — lectures/écritures DB + réseau)
# ---------------------------------------------------------------------------

def run_backtest(db_path: str | Path = _DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Pipeline complet de backtest :

    1. Charge tous les signaux historiques depuis scored_setups.
    2. Récupère 90 jours de prix OHLCV via yfinance pour chaque ticker.
    3. Persiste les prix frais dans daily_prices (étend l'historique existant).
    4. Charge la table daily_prices sans filtre de date.
    5. Calcule les retours T+5/T+10/T+20 pour chaque signal.
    6. Persiste les résultats dans backtest_results (idempotent).
    7. Retourne le DataFrame des résultats.

    Args:
        db_path: Chemin vers la base SQLite. Défaut : data/scanner.db.

    Returns:
        DataFrame au format backtest_results. Vide si scored_setups est vide
        (le scanner n'a jamais été exécuté avec --save).

    Effets de bord:
        Écrit dans daily_prices et backtest_results.
    """
    from src.config import BACKTEST_OFFSETS, BACKTEST_PRICE_LOOKBACK_DAYS
    from src.ingestion.market_data import fetch_ohlcv
    from src.storage.database import (
        initialize_db,
        load_all_daily_prices,
        load_all_scored_setups,
        upsert_backtest_results,
        upsert_daily_prices,
    )

    initialize_db(db_path)

    # 1. Charger tous les signaux historiques
    signals_df = load_all_scored_setups(db_path)
    if signals_df.empty:
        logger.warning(
            "run_backtest: scored_setups est vide — "
            "exécutez le scanner avec --save pendant au moins 1 jour."
        )
        return pd.DataFrame()

    tickers = signals_df["ticker"].dropna().unique().tolist()
    logger.info(
        "run_backtest: %d signaux, %d tickers uniques",
        len(signals_df), len(tickers),
    )

    # 2. Récupérer des prix frais sur 90 jours
    logger.info(
        "Récupération OHLCV %dj pour %d tickers via yfinance...",
        BACKTEST_PRICE_LOOKBACK_DAYS, len(tickers),
    )
    fresh_ohlcv = fetch_ohlcv(tickers, days=BACKTEST_PRICE_LOOKBACK_DAYS)

    # 3. Persister les prix frais (étend l'historique existant)
    if not fresh_ohlcv.empty:
        upsert_daily_prices(fresh_ohlcv, db_path)
        logger.info("Stocké %d lignes de prix frais", len(fresh_ohlcv))
    else:
        logger.warning("run_backtest: aucun prix récupéré depuis yfinance")

    # 4. Charger l'historique complet (sans filtre de date)
    prices_df = load_all_daily_prices(tickers, db_path)

    # 5. Calculer les retours
    results_df = _compute_returns(signals_df, prices_df, BACKTEST_OFFSETS)

    if results_df.empty:
        logger.warning("run_backtest: _compute_returns a retourné un DataFrame vide")
        return pd.DataFrame()

    # 6. Persister les résultats (INSERT OR REPLACE — idempotent)
    n_written = upsert_backtest_results(results_df, db_path)
    logger.info("run_backtest: %d lignes écrites dans backtest_results", n_written)

    return results_df


# ---------------------------------------------------------------------------
# Rapport de performance (fonction pure)
# ---------------------------------------------------------------------------

def _fmt_pct(val) -> str:
    """Formate un float en '+3.2%' / '-1.8%' / 'n/a' si None/NaN."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "n/a"
    return f"{val:+.1f}%"


def _win_rate(series: pd.Series) -> str:
    """Retourne 'XX.X%' de win rate (return > 0), ou 'n/a' si série vide."""
    valid = series.dropna()
    if valid.empty:
        return "n/a"
    return f"{(valid > 0).sum() / len(valid) * 100:.1f}%"


def _avg_ret(series: pd.Series) -> str:
    """Retourne '+X.X%' moyenne de la série, ou 'n/a' si vide."""
    valid = series.dropna()
    if valid.empty:
        return "n/a"
    return _fmt_pct(valid.mean())


def _performer_table(subset: pd.DataFrame, title: str) -> list[str]:
    """Formate une table TOP/WORST N performers."""
    out = ["", title]
    out.append(
        f"  {'Ticker':<8} {'Date':<12} {'Grade':<6} {'Flags':<16} "
        f"{'T+5':>6}  {'T+10':>6}  {'T+20':>7}"
    )
    for _, row in subset.iterrows():
        flags = " ".join(
            lbl
            for col, lbl in [
                ("accumulation_flag",   "ACCUM"),
                ("pre_breakout_flag",   "PRE-BRK"),
                ("cluster_buying_flag", "CLST"),
            ]
            if row.get(col, 0) == 1
        ) or "-"
        out.append(
            f"  {str(row['ticker']):<8} {str(row['signal_date']):<12} "
            f"{str(row.get('grade', '')):<6} {flags:<16} "
            f"{_fmt_pct(row.get('return_5d')):>6}  "
            f"{_fmt_pct(row.get('return_10d')):>6}  "
            f"{_fmt_pct(row.get('return_20d')):>7}"
        )
    return out


def generate_performance_report(backtest_df: pd.DataFrame) -> str:
    """
    Formate un rapport de performance multi-sections depuis les résultats
    du backtest.

    Sections :
      - OVERALL  : win rate et retour moyen à T+5/T+10/T+20
      - BY GRADE : agrégats par grade (A+, A, B, C)
      - BY FLAG  : agrégats par type de flag (ACCUM, PRE-BRK, CLST)
      - TOP 5 / WORST 5 performers (classés par return_20d)

    Args:
        backtest_df : DataFrame retourné par run_backtest() ou chargé depuis
                      load_backtest_results(). Colonnes attendues :
                      signal_date, ticker, grade, accumulation_flag,
                      pre_breakout_flag, cluster_buying_flag,
                      return_5d, return_10d, return_20d.

    Returns:
        Chaîne multi-lignes prête pour print(). Retourne un message "No data"
        si backtest_df est vide ou ne contient aucune donnée de retour.
    """
    today_str = date.today().strftime("%Y-%m-%d")
    sep = "=" * 65

    if backtest_df.empty:
        return (
            f"\n{sep}\n"
            "  RAPPORT DE PERFORMANCE DES SIGNAUX\n"
            "  Aucun signal historique trouvé — exécutez le scanner\n"
            "  avec --save pendant au moins 5 jours, puis relancez --backtest.\n"
            f"{sep}\n"
        )

    n_signals = len(backtest_df)
    n_tickers = backtest_df["ticker"].nunique()

    lines: list[str] = [
        "",
        sep,
        "  RAPPORT DE PERFORMANCE DES SIGNAUX",
        f"  Basé sur {n_signals} signaux issus de {n_tickers} tickers",
        f"  Généré le : {today_str}",
        sep,
    ]

    # Sous-ensembles par horizon disponible
    df_t5  = backtest_df.dropna(subset=["return_5d"])
    df_t10 = backtest_df.dropna(subset=["return_10d"])
    df_t20 = backtest_df.dropna(subset=["return_20d"])

    # --- OVERALL ---
    lines += [
        "",
        "GLOBAL (signaux avec données disponibles)",
        f"  Signaux évalués T+5  : {len(df_t5)}",
        f"  Signaux évalués T+10 : {len(df_t10)}",
        f"  Signaux évalués T+20 : {len(df_t20)}",
        f"  Win rate  T+5        : {_win_rate(df_t5['return_5d'])}",
        f"  Win rate  T+10       : {_win_rate(df_t10['return_10d'])}",
        f"  Win rate  T+20       : {_win_rate(df_t20['return_20d'])}",
        f"  Retour moy. T+5      : {_avg_ret(df_t5['return_5d'])}",
        f"  Retour moy. T+10     : {_avg_ret(df_t10['return_10d'])}",
        f"  Retour moy. T+20     : {_avg_ret(df_t20['return_20d'])}",
    ]

    # --- PAR GRADE ---
    lines += ["", "PAR GRADE"]
    lines.append(
        f"  {'Grade':<6} {'N':>4}  {'WR-T5':>7}  {'WR-T10':>7}  {'WR-T20':>7}  "
        f"{'Ret-T5':>8}  {'Ret-T10':>8}  {'Ret-T20':>8}"
    )
    for grade in ["A+", "A", "B", "C"]:
        sub20 = df_t20[df_t20["grade"] == grade]
        sub10 = df_t10[df_t10["grade"] == grade]
        sub5  = df_t5[df_t5["grade"] == grade]
        n = len(sub20)
        if n == 0:
            continue
        lines.append(
            f"  {grade:<6} {n:>4}  "
            f"{_win_rate(sub5['return_5d']):>7}  "
            f"{_win_rate(sub10['return_10d']):>7}  "
            f"{_win_rate(sub20['return_20d']):>7}  "
            f"{_avg_ret(sub5['return_5d']):>8}  "
            f"{_avg_ret(sub10['return_10d']):>8}  "
            f"{_avg_ret(sub20['return_20d']):>8}"
        )

    # --- PAR TYPE DE FLAG ---
    lines += ["", "PAR TYPE DE FLAG"]
    lines.append(
        f"  {'Flag':<8} {'N':>4}  {'WR-T5':>7}  {'WR-T10':>7}  {'WR-T20':>7}  "
        f"{'Ret-T5':>8}  {'Ret-T10':>8}  {'Ret-T20':>8}"
    )
    for flag_col, label in [
        ("accumulation_flag",   "ACCUM"),
        ("pre_breakout_flag",   "PRE-BRK"),
        ("cluster_buying_flag", "CLST"),
    ]:
        sub20 = df_t20[df_t20[flag_col] == 1]
        sub10 = df_t10[df_t10[flag_col] == 1]
        sub5  = df_t5[df_t5[flag_col] == 1]
        n = len(sub20)
        if n == 0:
            continue
        lines.append(
            f"  {label:<8} {n:>4}  "
            f"{_win_rate(sub5['return_5d']):>7}  "
            f"{_win_rate(sub10['return_10d']):>7}  "
            f"{_win_rate(sub20['return_20d']):>7}  "
            f"{_avg_ret(sub5['return_5d']):>8}  "
            f"{_avg_ret(sub10['return_10d']):>8}  "
            f"{_avg_ret(sub20['return_20d']):>8}"
        )

    # --- TOP 5 / WORST 5 ---
    if not df_t20.empty:
        top5   = df_t20.nlargest(5, "return_20d")
        worst5 = df_t20.nsmallest(5, "return_20d")
        lines += _performer_table(top5,   "TOP 5 PERFORMERS (retour T+20)")
        lines += _performer_table(worst5, "WORST 5 PERFORMERS (retour T+20)")

    lines += ["", sep, ""]
    return "\n".join(lines)
