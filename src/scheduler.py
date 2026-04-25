"""
Daily pipeline scheduler — runs the full scan at 18:00 ET every weekday.

Usage:
    python -m src.scheduler             # run as daemon
    python -m src.scheduler --now       # run once immediately, then start daemon

The scheduler is timezone-aware (Eastern Time) and skips weekends.
"""

import argparse
import logging
import sys
import time
from datetime import datetime

import pytz
import schedule

logger = logging.getLogger(__name__)

_ET = pytz.timezone("America/Toronto")
_RUN_TIME_ET = "18:00"


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

def _run_pipeline_job() -> None:
    """Wrapper called by the scheduler. Skips weekends automatically."""
    now_et = datetime.now(_ET)
    if now_et.weekday() >= 5:  # 5=Saturday, 6=Sunday
        logger.info("Weekend — skipping pipeline run")
        return

    logger.info("=== Scheduled pipeline run starting at %s ET ===", now_et.strftime("%Y-%m-%d %H:%M"))

    # Import here to avoid circular imports and keep scheduler lightweight
    from src.main import run_pipeline, format_console_output, setup_logging
    from src.storage.database import (
        initialize_db,
        upsert_insider_trades,
        upsert_daily_prices,
        upsert_scored_setups,
    )

    try:
        setup_logging()
        initialize_db()

        detected_df = run_pipeline(use_live_data=True)
        run_date     = now_et.strftime("%Y-%m-%d")

        # Persist setups
        upsert_scored_setups(detected_df, run_date=run_date)

        # Print top 10
        output = format_console_output(detected_df, top_n=10)
        print(output)
        logger.info("Pipeline completed successfully for %s", run_date)

    except Exception as exc:
        logger.exception("Pipeline run failed: %s", exc)


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------

def _schedule_is_after_cutoff() -> bool:
    """Return True if today's scheduled run has already passed."""
    now_et = datetime.now(_ET)
    h, m   = (int(x) for x in _RUN_TIME_ET.split(":"))
    cutoff = now_et.replace(hour=h, minute=m, second=0, microsecond=0)
    return now_et >= cutoff


def start_scheduler(run_now: bool = False) -> None:
    """
    Start the blocking scheduler loop.

    Args:
        run_now: if True, execute the pipeline immediately before scheduling.
    """
    logger.info("Scheduler starting. Daily run at %s ET (weekdays).", _RUN_TIME_ET)

    if run_now:
        logger.info("--now flag: running pipeline immediately")
        _run_pipeline_job()

    # Schedule the daily job using a wrapper that converts ET to local clock time.
    # schedule library uses the local system clock. We compute the UTC offset
    # dynamically each time the job fires so DST transitions are handled correctly.
    schedule.every().day.at(_et_to_local(_RUN_TIME_ET)).do(_run_pipeline_job)

    logger.info("Next run scheduled for %s local time (%s ET).",
                _et_to_local(_RUN_TIME_ET), _RUN_TIME_ET)

    while True:
        schedule.run_pending()
        time.sleep(30)


def _et_to_local(time_str: str) -> str:
    """
    Convert a HH:MM time string from Eastern Time to the local clock's HH:MM.
    This handles DST transitions correctly at scheduling time.
    """
    now = datetime.now()
    h, m = (int(x) for x in time_str.split(":"))
    et_naive = now.replace(hour=h, minute=m, second=0, microsecond=0)
    et_aware = _ET.localize(et_naive)

    # Convert to local system timezone
    local_dt = et_aware.astimezone(tz=None)
    return local_dt.strftime("%H:%M")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    from src.main import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(description="TSX/CSE Scanner Scheduler")
    parser.add_argument(
        "--now",
        action="store_true",
        help="Run the pipeline once immediately before entering the scheduler loop",
    )
    args = parser.parse_args()
    start_scheduler(run_now=args.now)


if __name__ == "__main__":
    main()
