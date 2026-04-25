"""
Streamlit dashboard — TSXV/CSE Insider Accumulation Scanner
Launch: python -m streamlit run dashboard.py
"""

import sys
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

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_setups() -> pd.DataFrame:
    try:
        from src.storage.database import load_last_scored_setups
        return load_last_scored_setups()
    except Exception as e:
        st.error(f"Erreur chargement setups: {e}")
        return pd.DataFrame()


def load_backtest() -> pd.DataFrame:
    try:
        from src.storage.database import load_backtest_results
        return load_backtest_results()
    except Exception as e:
        st.error(f"Erreur chargement backtest: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Grade colours
# ---------------------------------------------------------------------------

_GRADE_COLOUR = {"A+": "🟢", "A": "🟡", "B": "🟠", "C": "⚪"}


def _badge(grade: str) -> str:
    return f"{_GRADE_COLOUR.get(grade, '')} {grade}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    st.sidebar.header("🔍 Filtres")

    grades = [g for g in ["A+", "A", "B", "C"] if g in df["grade"].values]
    sel_grades = st.sidebar.multiselect("Grade", grades, default=grades)

    min_score = st.sidebar.slider("Score minimum", 0, 100,
                                   int(df["composite_score"].min()) if not df.empty else 0)

    st.sidebar.markdown("**Signaux actifs**")
    f_accum  = st.sidebar.checkbox("ACCUM")
    f_prebrk = st.sidebar.checkbox("PRE-BRK")
    f_clst   = st.sidebar.checkbox("CLST")

    top_n = st.sidebar.number_input("Top N", 5, 50, 20, 5)

    mask = df["grade"].isin(sel_grades) & (df["composite_score"] >= min_score)
    if f_accum:  mask &= df["accumulation_flag"].astype(bool)
    if f_prebrk: mask &= df["pre_breakout_flag"].astype(bool)
    if f_clst:   mask &= df["cluster_buying_flag"].astype(bool)

    return df[mask].head(int(top_n))


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_setups(raw: pd.DataFrame, filtered: pd.DataFrame):
    st.header("📋 Top Setups")

    if raw.empty:
        st.info("Aucune donnée. Lance `python -m src.main --save` d'abord.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers scannés", len(raw))
    c2.metric("Affichés", len(filtered))
    c3.metric("Grade A+/A", int(raw["grade"].isin(["A+", "A"]).sum()))
    c4.metric("Score moyen (affiché)",
              f"{filtered['composite_score'].mean():.1f}" if not filtered.empty else "—")

    if filtered.empty:
        st.info("Aucun setup ne correspond aux filtres.")
        return

    # Build display table
    rows = []
    for _, r in filtered.iterrows():
        flags = " ".join(f for f, col in [
            ("ACCUM", "accumulation_flag"),
            ("PRE-BRK", "pre_breakout_flag"),
            ("CLST", "cluster_buying_flag"),
        ] if r.get(col, 0))

        insider_val = float(r.get("latest_value_cad") or 0)
        val_str = f"${insider_val/1000:.0f}k" if insider_val >= 1000 else f"${insider_val:.0f}"

        rows.append({
            "Grade": _badge(str(r.get("grade", ""))),
            "Ticker": str(r.get("ticker", "")),
            "Entreprise": str(r.get("issuer", "") or ""),
            "Score": f"{r.get('composite_score', 0):.1f}",
            "Signaux": flags or "—",
            "Dernier achat": f"{r.get('insider_name', '')} {val_str}",
            "Vol ratio": f"{r.get('volume_ratio_5d', 0):.2f}×",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def section_chart(filtered: pd.DataFrame):
    st.header("📊 Scores")

    if filtered.empty:
        st.info("Aucun setup à afficher.")
        return

    chart_df = filtered[["ticker", "composite_score", "grade"]].copy()
    chart_df = chart_df.sort_values("composite_score", ascending=False)
    chart_df.columns = ["Ticker", "Score", "Grade"]
    chart_df = chart_df.set_index("Ticker")

    st.bar_chart(chart_df["Score"])


def section_backtest(bt: pd.DataFrame):
    st.header("🔬 Backtest")

    if bt.empty:
        st.info(
            "Pas encore de données backtest. "
            "Lance le scanner avec `--save` pendant plusieurs jours, "
            "puis `python -m src.main --backtest`."
        )
        return

    df5  = bt.dropna(subset=["return_5d"])
    df10 = bt.dropna(subset=["return_10d"])
    df20 = bt.dropna(subset=["return_20d"])

    def wr(s):
        v = s.dropna()
        return f"{(v > 0).mean()*100:.1f}%" if len(v) else "—"

    def avg(s):
        v = s.dropna()
        return f"{v.mean():+.1f}%" if len(v) else "—"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Signaux évalués", len(bt))
    c2.metric("Win rate T+5",  wr(df5["return_5d"]))
    c3.metric("Win rate T+10", wr(df10["return_10d"]))
    c4.metric("Win rate T+20", wr(df20["return_20d"]))
    c5.metric("Retour moy. T+5",  avg(df5["return_5d"]))
    c6.metric("Retour moy. T+20", avg(df20["return_20d"]))

    st.subheader("Résultats récents")
    cols = [c for c in ["signal_date","ticker","grade","composite_score",
                         "return_5d","return_10d","return_20d"] if c in bt.columns]
    st.dataframe(bt[cols].head(50), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("📈 TSXV / CSE Insider Accumulation Scanner")

    setups_df   = load_setups()
    backtest_df = load_backtest()
    filtered_df = apply_filters(setups_df)

    tab1, tab2, tab3 = st.tabs(["🏆 Top Setups", "📊 Scores", "🔬 Backtest"])

    with tab1:
        section_setups(setups_df, filtered_df)
    with tab2:
        section_chart(filtered_df)
    with tab3:
        section_backtest(backtest_df)

    st.divider()
    if not setups_df.empty and "run_date" in setups_df.columns:
        st.caption(f"Dernier scan : {setups_df['run_date'].max()}")


main()
