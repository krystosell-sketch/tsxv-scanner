"""
Streamlit dashboard — TSXV/CSE Insider Accumulation Scanner
Source : data/latest.json (généré chaque soir par GitHub Actions)
URL    : https://tsxv-scanner.streamlit.app

Launch local : python -m streamlit run dashboard.py
"""

import json
import os
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
# Client Anthropic
# ---------------------------------------------------------------------------

def _get_api_client():
    """Retourne un client Anthropic ou None si la clé n'est pas configurée."""
    try:
        import anthropic as _anthropic
        # Streamlit Cloud : secrets TOML
        key = ""
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            pass
        # Fallback : variable d'environnement / .env local
        key = key or os.getenv("ANTHROPIC_API_KEY", "")
        if key:
            return _anthropic.Anthropic(api_key=key)
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tv_url(ticker: str) -> str:
    """Construit l'URL TradingView pour un ticker TSXV/CSE."""
    clean    = ticker.replace(".CN", "").replace(".V", "")
    exchange = "CSE" if ticker.endswith(".CN") else "TSXV"
    return f"https://www.tradingview.com/chart/?symbol={exchange}:{clean}"


def _tsx_url(ticker: str) -> str:
    """Construit l'URL TSXTracker pour un ticker TSXV/CSE."""
    clean    = ticker.replace(".CN", "").replace(".V", "")
    exchange = "CSE" if ticker.endswith(".CN") else "TSXV"
    return f"https://tsxtracker.com/en/{exchange}/{clean}"


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
            "Ticker":        _tv_url(ticker),          # URL TradingView pour LinkColumn
            "TSXTracker":    _tsx_url(ticker),         # URL TSXTracker pour LinkColumn
            "Ticker_label":  ticker,                   # label affiché / clé de sélection
            "Entreprise":    str(r.get("issuer") or ""),
            "Score":         f"{float(r.get('composite_score', 0)):.1f}",
            "Signaux":       _fmt_flags(r),
            "Dernier achat": _fmt_insider(r),
            "Vol ratio":     f"{float(r.get('volume_ratio_5d', 0)):.2f}×",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tableau avec checkboxes
# ---------------------------------------------------------------------------

def _show_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Affiche le tableau avec une colonne checkbox de sélection.
    Retourne un sous-DataFrame des lignes cochées (depuis le df source original).
    """
    if df.empty:
        st.info(f"Aucun ticker en {label} pour le moment.")
        return pd.DataFrame()

    display = _build_display_df(df)
    display.insert(0, "✅", False)   # colonne checkbox en premier

    edited = st.data_editor(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "✅": st.column_config.CheckboxColumn("", width="small"),
            "Ticker": st.column_config.LinkColumn(
                label="📈 TV",
                display_text=r"symbol=\w+:(.+)$",
                width="small",
            ),
            "TSXTracker": st.column_config.LinkColumn(
                label="🔍 TSX",
                display_text=r"/(\w+)$",
                width="small",
            ),
            "Grade":        st.column_config.TextColumn("Grade", width="small"),
            "Ticker_label": None,                      # caché
            "Score":        st.column_config.TextColumn("Score", width="small"),
            "Vol ratio":    st.column_config.TextColumn("Vol", width="small"),
        },
        column_order=["✅", "Grade", "Ticker", "TSXTracker", "Entreprise", "Score", "Signaux", "Dernier achat", "Vol ratio"],
        disabled=["Grade", "Ticker", "TSXTracker", "Entreprise", "Score", "Signaux", "Dernier achat", "Vol ratio"],
    )

    selected_labels = edited.loc[edited["✅"] == True, "Ticker_label"].tolist()
    return df[df["ticker"].isin(selected_labels)].copy()


# ---------------------------------------------------------------------------
# Analyse Claude AI à la demande
# ---------------------------------------------------------------------------

def _show_ai_analysis(selected_df: pd.DataFrame) -> None:
    """
    Affiche le bouton d'analyse Claude AI pour les tickers sélectionnés.
    Les résultats s'affichent en cartes dépliables sous le tableau.
    """
    if selected_df.empty:
        return

    n = len(selected_df)
    st.markdown(f"*{n} ticker(s) sélectionné(s)*")

    if st.button(f"🤖 Analyser avec Claude AI ({n})", type="primary"):
        client = _get_api_client()
        if client is None:
            st.error(
                "❌ Clé API Anthropic non configurée.\n\n"
                "Sur Streamlit Cloud : ⚙️ Settings → Secrets → ajouter `ANTHROPIC_API_KEY`\n\n"
                "En local : ajouter `ANTHROPIC_API_KEY=sk-ant-...` dans le fichier `.env`"
            )
            return

        from src.ai.explainer import generate_explanation

        st.divider()
        for _, row in selected_df.iterrows():
            ticker = str(row.get("ticker", ""))
            grade  = str(row.get("grade", ""))
            issuer = str(row.get("issuer") or "").strip()

            with st.spinner(f"Analyse de {ticker} en cours..."):
                explanation = generate_explanation(row.to_dict(), client=client)

            label = f"**{ticker}**"
            if issuer and issuer not in ("nan", "None"):
                label += f" — {issuer}"
            label += f"  {_GRADE_EMOJI.get(grade, '')} Grade {grade}"

            with st.expander(label, expanded=True):
                st.markdown(explanation)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_attente(df: pd.DataFrame) -> None:
    """Tickers en phase Attente : ACCUM actif, breakout pas encore déclenché."""
    st.header("⏳ Attente")
    st.caption(
        "Accumulation en cours — cochez les tickers intéressants pour lancer l'analyse Claude AI. "
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

    selected = _show_table(attente, "Attente")
    _show_ai_analysis(selected)


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

    selected = _show_table(setup, "Setup")
    _show_ai_analysis(selected)


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
