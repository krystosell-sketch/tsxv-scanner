"""
Streamlit dashboard — TSXV/CSE Insider Accumulation Scanner
Source : data/latest.json (généré chaque soir par GitHub Actions)
URL    : https://tsxv-scanner.streamlit.app

Launch local : python -m streamlit run dashboard.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="TSXV/CSE Insider Scanner",
    page_icon="📈",
    layout="wide",
)

_LATEST_JSON = ROOT / "data" / "latest.json"

_GRADE_EMOJI = {"A+": "🟢", "A": "🟡", "B": "🟠", "C": "⚪"}


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, str]:
    """
    Charge data/latest.json et retourne (DataFrame, scan_date).
    Retourne un DataFrame vide si le fichier est absent.
    """
    if not _LATEST_JSON.exists():
        return pd.DataFrame(), ""

    try:
        with open(_LATEST_JSON, encoding="utf-8") as f:
            raw = json.load(f)
        df = pd.DataFrame(raw.get("tickers", []))
        scan_date = raw.get("scan_date", "")
        return df, scan_date
    except Exception as e:
        st.error(f"Erreur lecture latest.json : {e}")
        return pd.DataFrame(), ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tv_url(ticker: str) -> str:
    """Construit l'URL TradingView pour un ticker TSXV/CSE."""
    clean    = ticker.replace(".CN", "").replace(".V", "")
    exchange = "CSE" if ticker.endswith(".CN") else "TSXV"
    return f"https://www.tradingview.com/chart/?symbol={exchange}:{clean}"


def _fmt_insider(row: pd.Series) -> str:
    name  = str(row.get("insider_name") or "").strip()
    role  = str(row.get("latest_role") or "").strip()
    value = float(row.get("latest_value_cad") or 0)

    if not name or name in ("", "nan", "None"):
        return "—"

    role_str  = f" ({role})" if role else ""
    value_str = f"${value/1000:.0f}k" if value >= 1000 else f"${value:.0f}"
    return f"{name}{role_str} · {value_str}"


def _fmt_flags(row: pd.Series) -> str:
    parts = []
    if row.get("accumulation_flag"):   parts.append("ACCUM")
    if row.get("pre_breakout_flag"):   parts.append("PRE-BRK")
    if row.get("cluster_buying_flag"): parts.append("CLST")
    return " · ".join(parts) if parts else "—"


def _build_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Construit le DataFrame d'affichage avec toutes les colonnes formatées."""
    rows = []
    for _, r in df.iterrows():
        ticker = str(r.get("ticker", ""))
        grade  = str(r.get("grade", ""))
        rows.append({
            "Grade":         f"{_GRADE_EMOJI.get(grade, '')} {grade}",
            "Ticker":        _tv_url(ticker),          # URL pour LinkColumn
            "Ticker_label":  ticker,                   # label affiché
            "Entreprise":    str(r.get("issuer") or ""),
            "Score":         f"{float(r.get('composite_score', 0)):.1f}",
            "Signaux":       _fmt_flags(r),
            "Dernier achat": _fmt_insider(r),
            "Vol ratio":     f"{float(r.get('volume_ratio_5d', 0)):.2f}×",
        })
    return pd.DataFrame(rows)


def _show_table(df: pd.DataFrame, label: str) -> None:
    """Affiche un tableau avec tickers cliquables TradingView."""
    if df.empty:
        st.info(f"Aucun ticker en {label} pour le moment.")
        return

    display = _build_display_df(df)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.LinkColumn(
                label="Ticker",
                display_text=r"symbol=\w+:(.+)$",   # affiche seulement le symbole
            ),
            "Grade":         st.column_config.TextColumn("Grade", width="small"),
            "Ticker_label":  None,                   # caché (redondant avec Ticker)
            "Score":         st.column_config.TextColumn("Score", width="small"),
            "Vol ratio":     st.column_config.TextColumn("Vol", width="small"),
        },
        column_order=["Grade", "Ticker", "Entreprise", "Score", "Signaux", "Dernier achat", "Vol ratio"],
    )


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_attente(df: pd.DataFrame) -> None:
    """Tickers en phase Attente : ACCUM actif, breakout pas encore déclenché."""
    st.header("⏳ Attente")
    st.caption(
        "Accumulation en cours — l'analyse Claude AI a été faite sur ces tickers. "
        "Le breakout n'a pas encore eu lieu. Une alerte Discord sera envoyée dès qu'il se déclenche."
    )

    if df.empty:
        _show_table(df, "Attente")
        return

    attente = df[
        df["accumulation_flag"].astype(bool) &
        ~df["pre_breakout_flag"].astype(bool)
    ].copy()

    col1, col2 = st.columns(2)
    col1.metric("Tickers en Attente", len(attente))
    col2.metric("Grade A+/A", int(attente["grade"].isin(["A+", "A"]).sum()) if not attente.empty else 0)

    _show_table(attente, "Attente")


def section_setup(df: pd.DataFrame) -> None:
    """Tickers en phase Setup : breakout déclenché (PRE-BRK actif)."""
    st.header("🎯 Setup")
    st.caption(
        "Breakout déclenché — conditions de volume et de prix réunies. "
        "Tu as reçu une alerte Discord 🚨 au moment du déclenchement."
    )

    if df.empty:
        _show_table(df, "Setup")
        return

    setup = df[df["pre_breakout_flag"].astype(bool)].copy()

    col1, col2 = st.columns(2)
    col1.metric("Tickers en Setup", len(setup))
    col2.metric("Grade A+/A", int(setup["grade"].isin(["A+", "A"]).sum()) if not setup.empty else 0)

    _show_table(setup, "Setup")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📈 TSXV / CSE Insider Accumulation Scanner")

    df, scan_date = load_data()

    if df.empty:
        st.warning(
            "Aucune donnée disponible. "
            "Le scan de fin de journée se fait automatiquement chaque soir à 16h30 ET (jours ouvrables). "
            "En local, lance : `python -m src.main --mock --export-json`"
        )
        return

    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tickers scannés", len(df))
    col2.metric("En Attente",  int((df["accumulation_flag"].astype(bool) & ~df["pre_breakout_flag"].astype(bool)).sum()))
    col3.metric("En Setup",    int(df["pre_breakout_flag"].astype(bool).sum()))
    col4.metric("Grade A+/A", int(df["grade"].isin(["A+", "A"]).sum()))

    st.divider()

    tab1, tab2 = st.tabs(["⏳ Attente", "🎯 Setup"])

    with tab1:
        section_attente(df)

    with tab2:
        section_setup(df)

    st.divider()
    if scan_date:
        st.caption(f"Dernier scan : {scan_date} · Mise à jour chaque soir à 16h30 ET")


main()
