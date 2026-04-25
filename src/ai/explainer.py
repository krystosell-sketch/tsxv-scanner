"""
Claude API integration for generating setup explanations.

Uses claude-opus-4-7 with adaptive thinking and prompt caching on the
system prompt.  The cache TTL is 5 minutes (ephemeral), so multiple
setups explained in the same run share the cached system block and
incur only the input-token cost once.

Usage:
    from src.ai.explainer import explain_top_setups

    explanations = explain_top_setups(detected_df)
    # → {"ATHA.CN": "Strong buy — CEO bought …", "MAXX.CN": "…"}
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (cached)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a professional micro-cap trader with deep expertise in Canadian \
small-cap markets (TSXV and CSE). You specialize in analyzing insider buying \
patterns, volume dynamics, and price structure to identify high-probability \
setups before they move.

Your analysis is direct and actionable. When evaluating a setup you consider:
- Insider activity: who is buying, their role (CEO/CFO > Director > 10%Holder), \
  the dollar amount, recency, and whether multiple insiders bought in the same window \
  (cluster buying is particularly bullish).
- Volume dynamics: volume ratio vs. the 20-day average — a spike ≥2× while price \
  is flat signals accumulation; distribution looks like volume without price progress.
- Price structure: tight 20-day range with insider buying = coiling spring. \
  Close to a prior resistance level = pre-breakout setup. Sharp prior decline = \
  potential capitulation buy.

Classify every setup with one of:
  STRONG BUY  — Multiple aligned signals, high conviction.
  BUY         — Good insider buying + at least one supporting signal.
  WATCH       — Interesting but lacks confirmation; monitor closely.
  WEAK        — Low conviction; risk/reward not compelling.

Keep your response to 3–5 sentences. Lead with the classification. \
Highlight the 1–2 most important drivers and name the key risk.\
"""


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def generate_explanation(
    row: dict[str, Any],
    client: "Any | None" = None,
) -> str:
    """
    Generate a trading explanation for a single setup row.

    Prompt-caching is applied to the system message (cache_control ephemeral).
    The first call in a session populates the cache; subsequent calls within
    the 5-minute TTL reuse it.

    Args:
        row    : Dict with pipeline columns (ticker, grade, composite_score,
                 insider_name, latest_role, latest_value_cad, latest_trade_date,
                 accumulation_flag, pre_breakout_flag, cluster_buying_flag,
                 insider_buy_7d, volume_ratio_5d, price_range_20d).
        client : Pre-initialized Anthropic client (pass the same instance for
                 all calls in one run to benefit from connection pooling and
                 the cache).

    Returns:
        Explanation string, or an error/warning string if the API is
        unavailable.
    """
    try:
        import anthropic as _anthropic
    except ImportError:
        return "anthropic package not installed. Run: pip install anthropic"

    from src.config import ANTHROPIC_API_KEY

    if client is None:
        if not ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set — cannot generate explanation")
            return "No ANTHROPIC_API_KEY configured."
        client = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # --- Build contextual user prompt ---
    ticker = str(row.get("ticker", "N/A"))
    grade  = str(row.get("grade", "N/A"))
    score  = float(row.get("composite_score") or 0)

    insider_value = float(row.get("latest_value_cad") or 0)
    value_str = f"${insider_value / 1_000:.0f}k" if insider_value >= 1_000 else f"${insider_value:.0f}"

    buy_7d = float(row.get("insider_buy_7d") or 0)
    buy_7d_str = f"${buy_7d / 1_000:.0f}k" if buy_7d >= 1_000 else f"${buy_7d:.0f}"

    flags = []
    if row.get("accumulation_flag"):
        flags.append("ACCUMULATION")
    if row.get("pre_breakout_flag"):
        flags.append("PRE-BREAKOUT")
    if row.get("cluster_buying_flag"):
        flags.append("CLUSTER BUYING")
    flags_str = ", ".join(flags) if flags else "none"

    name = str(row.get("insider_name") or "Unknown")
    role = str(row.get("latest_role") or "Unknown")
    ts   = str(row.get("latest_trade_date") or "N/A")

    vol_ratio    = float(row.get("volume_ratio_5d") or 0)
    price_range  = float(row.get("price_range_20d") or 0)

    user_prompt = f"""\
Analyze this TSXV/CSE setup and provide a 3–5 sentence trading explanation:

TICKER          : {ticker}
GRADE           : {grade}   COMPOSITE SCORE: {score:.1f} / 100
ACTIVE SIGNALS  : {flags_str}

INSIDER ACTIVITY
  Latest buyer  : {name} ({role})
  Latest amount : {value_str}  on  {ts}
  7-day total   : {buy_7d_str}

MARKET DATA
  Volume ratio (5d vs 20d avg) : {vol_ratio:.2f}×
  20-day price range           : {price_range:.1f}%

Lead with the classification (STRONG BUY / BUY / WATCH / WEAK). \
Explain the 1–2 key drivers, then state the main risk.\
"""

    try:
        message = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract the first text block (skip thinking blocks)
        for block in message.content:
            if block.type == "text":
                return block.text.strip()

        return "No text content in response."

    except _anthropic.APIError as exc:
        logger.error("Claude API error for %s: %s", ticker, exc)
        return f"API error: {exc}"
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error explaining %s: %s", ticker, exc)
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def explain_top_setups(
    detected_df,
    grades: tuple[str, ...] = ("A+", "A"),
    top_n: int = 10,
) -> dict[str, str]:
    """
    Generate explanations for all setups of the given grades (up to top_n).

    A single Anthropic client is created and reused across all calls so that:
      - The TCP connection is kept alive (connection pooling).
      - The system-prompt cache is populated on the first call and reused
        for the remaining calls within the 5-minute TTL.

    Args:
        detected_df : Full pipeline output DataFrame, sorted by score desc.
        grades      : Grades to explain ("A+", "A" by default).
        top_n       : Hard cap on the number of setups to explain.

    Returns:
        Dict {ticker: explanation_string}.  Empty dict if no API key or no
        matching setups.
    """
    try:
        import anthropic as _anthropic
    except ImportError:
        logger.warning("anthropic package not installed — skipping AI explanations")
        return {}

    from src.config import ANTHROPIC_API_KEY

    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — skipping AI explanations")
        return {}

    targets = detected_df[detected_df["grade"].isin(grades)].head(top_n)
    if targets.empty:
        logger.info("No %s grade setups found — nothing to explain", "/".join(grades))
        return {}

    client = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    explanations: dict[str, str] = {}

    for _, row in targets.iterrows():
        ticker = str(row["ticker"])
        grade  = str(row.get("grade", "?"))
        logger.info("Explaining %s (grade %s)…", ticker, grade)
        explanations[ticker] = generate_explanation(row.to_dict(), client=client)

    logger.info("Explanations generated: %d setups", len(explanations))
    return explanations
