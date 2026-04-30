"""
Génère un graphique matplotlib des top setups et retourne les bytes PNG.
Envoyé comme pièce jointe dans l'alerte Discord.
"""

import io
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def generate_score_chart(detected_df: pd.DataFrame, top_n: int = 10) -> bytes | None:
    """
    Génère un graphique horizontal des scores composites pour les top N setups.

    Args:
        detected_df : DataFrame pipeline trié par score desc.
        top_n       : Nombre de tickers à afficher.

    Returns:
        Bytes PNG du graphique, ou None si erreur.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Pas d'affichage GUI
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        top = detected_df.head(top_n).copy()
        if top.empty:
            return None

        # Couleurs par grade
        colour_map = {"A+": "#00C853", "A": "#76FF03", "B": "#FFD600", "C": "#9E9E9E"}
        colours = [colour_map.get(str(g), "#9E9E9E") for g in top["grade"]]

        # Labels : ticker + nom entreprise si disponible
        labels = []
        for _, row in top.iterrows():
            ticker = str(row["ticker"])
            issuer = str(row.get("issuer") or "").strip()
            if issuer and issuer not in ("nan", "None", ""):
                labels.append(f"{ticker}\n{issuer[:25]}")
            else:
                labels.append(ticker)

        scores = top["composite_score"].tolist()

        # Figure
        fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.6)))
        fig.patch.set_facecolor("#1E1E2E")
        ax.set_facecolor("#1E1E2E")

        bars = ax.barh(range(len(labels)), scores, color=colours, height=0.6, edgecolor="none")

        # Valeurs sur les barres
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}", va="center", ha="left",
                color="white", fontsize=9, fontweight="bold",
            )

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, color="white", fontsize=9)
        ax.set_xlabel("Score composite", color="white", fontsize=10)
        ax.set_xlim(0, 105)
        ax.invert_yaxis()
        ax.tick_params(colors="white")
        ax.spines[:].set_visible(False)
        ax.xaxis.label.set_color("white")
        ax.tick_params(axis="x", colors="#888888")

        # Titre
        from datetime import date
        ax.set_title(
            f"TSXV/CSE Insider Scanner — {date.today().strftime('%Y-%m-%d')}",
            color="white", fontsize=12, fontweight="bold", pad=12,
        )

        # Légende grades
        legend_patches = [
            mpatches.Patch(color=c, label=f"Grade {g}")
            for g, c in colour_map.items()
        ]
        ax.legend(
            handles=legend_patches, loc="lower right",
            facecolor="#2E2E3E", edgecolor="none",
            labelcolor="white", fontsize=8,
        )

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except Exception as exc:
        logger.error("generate_score_chart error: %s", exc)
        return None
