"""
Central configuration: all thresholds, weights, and constants. No logic here.
"""

# Charge automatiquement le fichier .env s'il existe (sans erreur si absent)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv pas installé — les variables système sont utilisées directement

# ---------------------------------------------------------------------------
# API Keys — placeholders for Phase 2
# ---------------------------------------------------------------------------
ALPHA_VANTAGE_API_KEY: str = ""
SEDI_API_KEY: str = ""

# ---------------------------------------------------------------------------
# Score weights (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHT_INSIDER: float = 0.40
WEIGHT_VOLUME: float = 0.25
WEIGHT_PRICE: float = 0.20
WEIGHT_CATALYST: float = 0.15

# ---------------------------------------------------------------------------
# Grade thresholds
# ---------------------------------------------------------------------------
GRADE_A_PLUS: int = 80
GRADE_A: int = 60
GRADE_B: int = 40

# ---------------------------------------------------------------------------
# Insider score breakpoints (value in CAD)
# ---------------------------------------------------------------------------
INSIDER_BUY_7D_HIGH: int = 500_000     # → 40 pts
INSIDER_BUY_7D_MID: int = 100_000      # → 25 pts
INSIDER_BUY_7D_LOW: int = 50_000       # → 15 pts

INSIDER_CLUSTER_BONUS: int = 25
INSIDER_CEO_CFO_BONUS: int = 20
INSIDER_BUY_MKTCAP_BONUS: int = 15
INSIDER_BUY_MKTCAP_THRESHOLD: float = 0.01  # 1%

# ---------------------------------------------------------------------------
# Accumulation detection thresholds
# ---------------------------------------------------------------------------
ACCUM_MIN_BUY_7D_CAD: int = 50_000
ACCUM_MIN_BUY_COUNT_7D: int = 2
ACCUM_MIN_VOLUME_RATIO_5D: float = 1.3
ACCUM_MAX_PRICE_RANGE_20D_PCT: float = 15.0

# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------
VOLUME_SPIKE_MULTIPLIER: float = 2.0
ACCUM_VOLUME_MIN_DAYS_ABOVE_AVG: int = 3   # of last 5 days above 20d avg

# ---------------------------------------------------------------------------
# Price structure
# ---------------------------------------------------------------------------
NEAR_BREAKOUT_THRESHOLD: float = 0.90       # close >= low + 90% of range
POST_DROP_THRESHOLD: float = 0.20           # drop > 20% over 30 days

# ---------------------------------------------------------------------------
# Cluster buying
# ---------------------------------------------------------------------------
CLUSTER_WINDOW_DAYS: int = 14
CLUSTER_MIN_INSIDERS: int = 2

# ---------------------------------------------------------------------------
# Role weights
# ---------------------------------------------------------------------------
ROLE_WEIGHTS: dict[str, int] = {
    "CEO": 3,
    "CFO": 3,
    "Director": 2,
    "10%Holder": 1,
    "Other": 1,
}

# ---------------------------------------------------------------------------
# Mock data universe
# ---------------------------------------------------------------------------
MOCK_OHLCV_DAYS: int = 30

MOCK_TICKERS: list[str] = [
    # Strong accumulation (designed to score A+)
    "VLE.V", "NVNI.CN", "LITH.V", "PNPN.CN",
    # Near-breakout (designed to score A)
    "FIND.V", "CRTM.CN", "AVCR.V", "BLOK.CN",
    # Mild accumulation (designed to score B)
    "CSTR.V", "GCX.V", "LODE.CN", "MAGT.V", "SAFE.CN",
    # No signal (designed to score C)
    "IRON.V", "GLDN.CN", "SLVX.V", "AQUA.CN", "NRTH.V", "PINE.CN", "CBAL.V",
]

# ---------------------------------------------------------------------------
# Backtest configuration
# ---------------------------------------------------------------------------
BACKTEST_PRICE_LOOKBACK_DAYS: int = 90   # fenêtre yfinance lors du backtest
BACKTEST_OFFSETS: list[int] = [5, 10, 20]  # décalages calendaires T+N

# ---------------------------------------------------------------------------
# Phase 4 — External services
# Set these via environment variables or replace the defaults below.
# ---------------------------------------------------------------------------
import os

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
