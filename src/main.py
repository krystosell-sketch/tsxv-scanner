"""
Pipeline orchestrator.

Modes:
  python -m src.main           → live data (TSXTracker + yfinance)
  python -m src.main --mock    → Phase 1 mock data (no network)
  python -m src.main --save    → live data + save results to SQLite
"""

import argparse
import logging
import logging.handlers
import os
from datetime import date

import pandas as pd

from src.detection.accumulation import detect_signals
from src.processing.features import compute_features
from src.processing.normalize import normalize_insider_df, normalize_ohlcv_df
from src.processing.scoring import compute_scores

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str = "logs") -> None:
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:
        return  # already configured
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    fh = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "scanner.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(ch)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_mock() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    from src.ingestion.mock_data import (
        generate_insider_trades,
        generate_ohlcv,
        get_mock_market_caps,
    )
    return generate_insider_trades(), generate_ohlcv(), get_mock_market_caps()


def _load_live() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    from src.ingestion.insider_scraper import fetch_insider_trades
    from src.ingestion.market_data import fetch_ohlcv, fetch_market_caps

    # 1. Fetch and normalize insider data
    logger.info("Fetching live insider trades from TSXTracker...")
    raw_insider = fetch_insider_trades(lookback_days=45, max_pages=15)
    insider_df  = normalize_insider_df(raw_insider)

    if insider_df.empty:
        logger.warning("No insider trades found — falling back to mock data")
        return _load_mock()

    # 2. Derive the unique tickers to fetch OHLCV for
    tickers = insider_df["ticker"].dropna().unique().tolist()
    logger.info("Fetching OHLCV for %d tickers via yfinance...", len(tickers))
    raw_ohlcv = fetch_ohlcv(tickers, days=30)

    # 3. Market caps
    logger.info("Fetching market caps...")
    market_caps = fetch_market_caps(tickers)

    return raw_insider, raw_ohlcv, market_caps


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    use_live_data: bool = True,
    reference_date: pd.Timestamp | None = None,
    save_to_db: bool = False,
) -> pd.DataFrame:
    """
    Full pipeline. Returns detected_df sorted by composite_score desc.

    Args:
        use_live_data: if False, uses mock data (no network calls)
        reference_date: override the anchor date for feature windows
        save_to_db: if True, persist results to SQLite
    """
    logger.info("Pipeline start (live=%s)", use_live_data)

    # 1. Ingest
    if use_live_data:
        raw_insider, raw_ohlcv, market_caps = _load_live()
    else:
        raw_insider, raw_ohlcv, market_caps = _load_mock()

    logger.info("Ingestion complete: %d insider rows, %d ohlcv rows",
                len(raw_insider), len(raw_ohlcv))

    # 2. Normalize
    insider_df = normalize_insider_df(raw_insider)
    ohlcv_df   = normalize_ohlcv_df(raw_ohlcv)
    logger.info("Normalized: %d insider buys, %d ohlcv rows",
                len(insider_df), len(ohlcv_df))

    if ohlcv_df.empty:
        logger.error("No OHLCV data — cannot compute features")
        return pd.DataFrame()

    # 3. Features
    features_df = compute_features(insider_df, ohlcv_df, market_caps, reference_date)
    logger.info("Features: %d tickers", len(features_df))

    # 4. Score
    scored_df = compute_scores(features_df)

    # 5. Detect
    detected_df = detect_signals(scored_df)
    detected_df = detected_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    # 6. Persist (optional)
    if save_to_db and not detected_df.empty:
        from src.storage.database import (
            initialize_db, upsert_insider_trades,
            upsert_daily_prices, upsert_scored_setups,
        )
        run_date = date.today().isoformat()
        initialize_db()
        upsert_insider_trades(insider_df)
        upsert_daily_prices(ohlcv_df)
        upsert_scored_setups(detected_df, run_date=run_date)
        logger.info("Results saved to SQLite (run_date=%s)", run_date)

    logger.info("Pipeline complete — %d tickers scored", len(detected_df))
    return detected_df


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def format_console_output(detected_df: pd.DataFrame, top_n: int = 10) -> str:
    today    = date.today().strftime("%Y-%m-%d")
    universe = len(detected_df)
    top      = detected_df.head(top_n)

    sep  = "=" * 68
    dash = "-" * 68

    lines: list[str] = [
        sep,
        "  TSXV/CSE INSIDER ACCUMULATION SCANNER",
        f"  Run date: {today}  |  Universe: {universe} tickers",
        sep,
        "",
        f"{'Rank':>4}  {'Ticker':<10} {'Score':>5}  {'Grade':<5}  {'Flags':<17}  Latest Insider Activity",
        f"{'----':>4}  {'-'*10} {'-----':>5}  {'-----':<5}  {'-'*17}  {'-'*38}",
    ]

    for i, row in top.iterrows():
        rank     = int(i) + 1
        ticker   = str(row["ticker"])
        score    = row["composite_score"]
        grade    = row["grade"]
        flags    = _format_flags(row)
        activity = _format_activity(row)
        lines.append(
            f"{rank:>4}  {ticker:<10} {score:>5.1f}  {grade:<5}  {flags:<17}  {activity}"
        )

    lines += ["", dash, _format_summary(detected_df, top_n), sep]

    return "\n".join(lines)


def _format_flags(row: pd.Series) -> str:
    parts = []
    if row.get("accumulation_flag", False):
        parts.append("ACCUM")
    if row.get("pre_breakout_flag", False):
        parts.append("PRE-BRK")
    if row.get("cluster_buying_flag", False):
        parts.append("CLST")
    return " ".join(parts) if parts else "-"


def _format_activity(row: pd.Series) -> str:
    name  = str(row.get("insider_name", "") or "")
    role  = str(row.get("latest_role", "") or "")
    value = float(row.get("latest_value_cad", 0) or 0)
    ts    = row.get("latest_trade_date")

    if not name or name in ("", "nan"):
        return "No recent insider activity"

    role_str  = f" ({role})" if role else ""
    value_str = f"${value/1000:.0f}k" if value >= 1000 else f"${value:.0f}"

    date_str = ""
    if ts is not None and pd.notna(ts):
        try:
            date_str = f" [{pd.Timestamp(ts).strftime('%Y-%m-%d')}]"
        except Exception:
            pass

    return f"{name}{role_str} {value_str}{date_str}"


def _format_summary(df: pd.DataFrame, top_n: int) -> str:
    n_accum  = int(df["accumulation_flag"].sum())
    n_pre    = int(df["pre_breakout_flag"].sum())
    n_clst   = int(df["cluster_buying_flag"].sum())
    n_signal = int(df["grade"].isin(["A+", "A", "B"]).sum())
    n_active = int((df["insider_buy_30d"] > 0).sum())
    return (
        f"  Signals: {n_accum} ACCUM  |  {n_pre} PRE-BRK  |  {n_clst} CLST  |  "
        f"{n_signal} grades B+ or higher\n"
        f"  Tickers scanned: {len(df)}  |  With insider activity (30d): {n_active}"
    )


# ---------------------------------------------------------------------------
# JSON export (pour dashboard Streamlit Cloud)
# ---------------------------------------------------------------------------

def export_latest_json(detected_df: pd.DataFrame, path: str = "data/latest.json") -> str:
    """
    Exporte les résultats du pipeline dans data/latest.json pour le dashboard public.

    Colonnes exportées : ticker, grade, composite_score, accumulation_flag,
    pre_breakout_flag, cluster_buying_flag, issuer, insider_name, latest_role,
    latest_value_cad, latest_trade_date, volume_ratio_5d, signal_reasons.

    Returns:
        Chemin du fichier écrit.
    """
    import json
    import os

    cols = [
        "ticker", "grade", "composite_score",
        "accumulation_flag", "pre_breakout_flag", "cluster_buying_flag",
        "issuer", "insider_name", "latest_role",
        "latest_value_cad", "latest_trade_date",
        "volume_ratio_5d", "price_range_20d", "signal_reasons",
    ]
    # Garder seulement les colonnes présentes
    available = [c for c in cols if c in detected_df.columns]
    export_df = detected_df[available].copy()

    # Convertir les flags en bool Python natif pour JSON
    for flag in ("accumulation_flag", "pre_breakout_flag", "cluster_buying_flag"):
        if flag in export_df.columns:
            export_df[flag] = export_df[flag].astype(bool)

    # Convertir signal_reasons (liste ou str pipe-délimité) en liste
    if "signal_reasons" in export_df.columns:
        export_df["signal_reasons"] = export_df["signal_reasons"].apply(
            lambda x: x if isinstance(x, list)
            else [r for r in str(x).split("|") if r.strip()]
        )

    # Convertir les dates en str ISO
    if "latest_trade_date" in export_df.columns:
        export_df["latest_trade_date"] = export_df["latest_trade_date"].apply(
            lambda x: str(x)[:10] if pd.notna(x) and str(x) not in ("", "nan", "NaT") else ""
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    records = export_df.fillna("").to_dict(orient="records")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"scan_date": date.today().isoformat(), "tickers": records},
            f, ensure_ascii=False, indent=2,
        )

    logger.info("export_latest_json: %d tickers → %s", len(records), path)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="TSX/CSE Insider Accumulation Scanner")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock data instead of live scraping")
    parser.add_argument("--save", action="store_true",
                        help="Save results to SQLite database")
    parser.add_argument("--backtest", action="store_true",
                        help="Calcule les retours T+5/T+10/T+20 et affiche le rapport de performance")
    parser.add_argument("--explain", action="store_true",
                        help="Génère des explications Claude (API) pour les setups grade A+ et A")
    parser.add_argument("--alert", action="store_true",
                        help="Envoie les top setups du jour via webhook Discord")
    parser.add_argument("--export-json", action="store_true",
                        help="Exporte les resultats dans data/latest.json pour le dashboard")
    args = parser.parse_args()

    use_live = not args.mock
    detected_df = run_pipeline(use_live_data=use_live, save_to_db=args.save)

    if not detected_df.empty:
        print(format_console_output(detected_df, top_n=10))
    else:
        print("No results — check logs for errors.")

    if args.backtest:
        from src.backtest.backtest import run_backtest, generate_performance_report
        logger.info("Lancement du backtest...")
        backtest_df = run_backtest()
        print(generate_performance_report(backtest_df))

    # --- Phase 4: AI explanations + Discord alerts ---
    explanations: dict[str, str] = {}

    if args.explain and not detected_df.empty:
        from src.ai.explainer import explain_top_setups
        # Expliquer uniquement les tickers en phase Attente (ACCUM sans PRE-BRK)
        # L'analyse Claude aide à décider si ça vaut la peine de surveiller AVANT le breakout
        logger.info("Generation des explications Claude pour les setups en Attente (A+/A)...")
        explanations = explain_top_setups(
            detected_df,
            grades=("A+", "A"),
            attente_only=True,  # uniquement ACCUM sans PRE-BRK
        )
        if explanations:
            print("\n--- Explications AI (Attente — Grade A+ et A) ---\n")
            for ticker, explanation in explanations.items():
                row = detected_df[detected_df["ticker"] == ticker].iloc[0]
                print(f"{ticker} (Grade {row['grade']}, score {row['composite_score']:.1f}):")
                print(f"  {explanation}\n")
        else:
            print("\n[--explain] Aucun setup Attente A+/A, ou ANTHROPIC_API_KEY manquant.\n")

    if args.alert and not detected_df.empty:
        from src.alerts.discord import send_daily_alert
        logger.info("Envoi de l'alerte Discord...")
        # Filtrer comme le dashboard : seulement les tickers avec au moins un signal actif
        flagged_df = detected_df[
            detected_df["accumulation_flag"].astype(bool) |
            detected_df["pre_breakout_flag"].astype(bool) |
            detected_df["cluster_buying_flag"].astype(bool)
        ].copy()
        alert_df = flagged_df if not flagged_df.empty else detected_df
        success = send_daily_alert(alert_df, explanations=explanations, top_n=10)
        status = "[OK] Alerte Discord envoyee." if success else "[ERREUR] Echec de l'envoi Discord (verifiez DISCORD_WEBHOOK_URL)."
        print(status)

    if args.export_json and not detected_df.empty:
        path = export_latest_json(detected_df)
        print(f"[OK] Resultats exportes → {path}")


if __name__ == "__main__":
    main()
