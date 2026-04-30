"""
SQLite persistence layer.

Three tables:
  insider_trades  — raw cleaned buy transactions
  daily_prices    — OHLCV per ticker per date
  scored_setups   — pipeline output snapshot per run
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "scanner.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS insider_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_date     TEXT    NOT NULL,
    trade_date      TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    issuer          TEXT,
    insider_name    TEXT,
    role            TEXT,
    price_cad       REAL,
    quantity        INTEGER,
    value_cad       REAL,
    nb_owned_after  INTEGER,
    delta_own_pct   REAL,
    general_remarks TEXT,
    market_cap_cad  REAL,
    inserted_at     TEXT    DEFAULT (datetime('now')),
    UNIQUE (trade_date, ticker, insider_name, quantity)
);

CREATE TABLE IF NOT EXISTS daily_prices (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    date      TEXT    NOT NULL,
    ticker    TEXT    NOT NULL,
    open      REAL,
    high      REAL,
    low       REAL,
    close     REAL    NOT NULL,
    volume    INTEGER,
    UNIQUE (date, ticker)
);

CREATE TABLE IF NOT EXISTS scored_setups (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date                TEXT    NOT NULL,
    ticker                  TEXT    NOT NULL,
    composite_score         REAL,
    grade                   TEXT,
    insider_score           REAL,
    volume_score            REAL,
    price_structure_score   REAL,
    accumulation_flag       INTEGER,
    pre_breakout_flag       INTEGER,
    cluster_buying_flag     INTEGER,
    insider_buy_7d          REAL,
    volume_ratio_5d         REAL,
    price_range_20d         REAL,
    signal_reasons          TEXT,
    issuer                  TEXT,
    insider_name            TEXT,
    latest_role             TEXT,
    latest_value_cad        REAL,
    latest_trade_date       TEXT,
    UNIQUE (run_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_trades_ticker_date ON insider_trades (ticker, trade_date);
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON daily_prices (ticker, date);
CREATE INDEX IF NOT EXISTS idx_setups_run_date     ON scored_setups (run_date, composite_score DESC);

CREATE TABLE IF NOT EXISTS backtest_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_date         TEXT    NOT NULL,
    ticker              TEXT    NOT NULL,
    grade               TEXT,
    composite_score     REAL,
    accumulation_flag   INTEGER,
    pre_breakout_flag   INTEGER,
    cluster_buying_flag INTEGER,
    close_at_signal     REAL,
    close_t5            REAL,
    close_t10           REAL,
    close_t20           REAL,
    return_5d           REAL,
    return_10d          REAL,
    return_20d          REAL,
    computed_at         TEXT    DEFAULT (datetime('now')),
    UNIQUE (signal_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_backtest_ticker_date
    ON backtest_results (ticker, signal_date);
"""


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def get_connection(db_path: str | Path = _DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_db(db_path: str | Path = _DEFAULT_DB_PATH) -> None:
    """Create tables and indexes if they don't exist. Migrates existing DBs."""
    with get_connection(db_path) as conn:
        conn.executescript(_DDL)
        # Migration: add new columns to scored_setups if they don't exist yet
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(scored_setups)").fetchall()
        }
        migrations = {
            "issuer":           "ALTER TABLE scored_setups ADD COLUMN issuer TEXT",
            "insider_name":     "ALTER TABLE scored_setups ADD COLUMN insider_name TEXT",
            "latest_role":      "ALTER TABLE scored_setups ADD COLUMN latest_role TEXT",
            "latest_value_cad": "ALTER TABLE scored_setups ADD COLUMN latest_value_cad REAL",
            "latest_trade_date":"ALTER TABLE scored_setups ADD COLUMN latest_trade_date TEXT",
        }
        for col, ddl in migrations.items():
            if col not in existing:
                conn.execute(ddl)
                logger.info("Migration: added column scored_setups.%s", col)
        conn.commit()
    logger.info("Database initialized: %s", db_path)


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------

def upsert_insider_trades(df: pd.DataFrame, db_path: str | Path = _DEFAULT_DB_PATH) -> int:
    """
    Insert new insider trade rows; skip duplicates (same trade_date/ticker/name/qty).
    Returns the number of rows inserted.
    """
    if df.empty:
        return 0

    cols = [
        "filing_date", "trade_date", "ticker", "issuer", "insider_name",
        "role", "price_cad", "quantity", "value_cad", "nb_owned_after",
        "delta_own_pct", "general_remarks", "market_cap_cad",
    ]
    insert_df = _prep(df, cols)

    sql = f"""
        INSERT OR IGNORE INTO insider_trades
            ({', '.join(cols)})
        VALUES
            ({', '.join(['?'] * len(cols))})
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(sql, insert_df.itertuples(index=False, name=None))
        inserted = cursor.rowcount if cursor.rowcount >= 0 else 0
        conn.commit()

    logger.info("upsert_insider_trades: inserted ~%d rows", len(insert_df))
    return len(insert_df)


def upsert_daily_prices(df: pd.DataFrame, db_path: str | Path = _DEFAULT_DB_PATH) -> int:
    """
    Insert/replace daily price rows. Returns rows processed.
    """
    if df.empty:
        return 0

    cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    insert_df = _prep(df, cols)

    sql = f"""
        INSERT OR REPLACE INTO daily_prices
            ({', '.join(cols)})
        VALUES
            ({', '.join(['?'] * len(cols))})
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(sql, insert_df.itertuples(index=False, name=None))
        conn.commit()

    logger.info("upsert_daily_prices: %d rows", len(insert_df))
    return len(insert_df)


def upsert_scored_setups(df: pd.DataFrame, run_date: str, db_path: str | Path = _DEFAULT_DB_PATH) -> int:
    """
    Insert/replace today's scored setups snapshot.
    """
    if df.empty:
        return 0

    cols = [
        "run_date", "ticker", "composite_score", "grade",
        "insider_score", "volume_score", "price_structure_score",
        "accumulation_flag", "pre_breakout_flag", "cluster_buying_flag",
        "insider_buy_7d", "volume_ratio_5d", "price_range_20d",
        "signal_reasons",
        "issuer", "insider_name", "latest_role", "latest_value_cad", "latest_trade_date",
    ]

    out = df.copy()
    out["run_date"] = run_date
    # Convert list column to pipe-delimited string
    if "signal_reasons" in out.columns:
        out["signal_reasons"] = out["signal_reasons"].apply(
            lambda x: "|".join(x) if isinstance(x, list) else str(x or "")
        )
    for flag_col in ("accumulation_flag", "pre_breakout_flag", "cluster_buying_flag"):
        if flag_col in out.columns:
            out[flag_col] = out[flag_col].astype(int)
    # Convert trade date to ISO string (datetime → "YYYY-MM-DD")
    if "latest_trade_date" in out.columns:
        out["latest_trade_date"] = pd.to_datetime(
            out["latest_trade_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d").fillna("")

    insert_df = _prep(out, cols)
    sql = f"""
        INSERT OR REPLACE INTO scored_setups
            ({', '.join(cols)})
        VALUES
            ({', '.join(['?'] * len(cols))})
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(sql, insert_df.itertuples(index=False, name=None))
        conn.commit()

    logger.info("upsert_scored_setups: %d rows for run_date=%s", len(insert_df), run_date)
    return len(insert_df)


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def load_insider_trades(
    lookback_days: int = 45,
    db_path: str | Path = _DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Load insider trades from the last N days."""
    sql = """
        SELECT * FROM insider_trades
        WHERE trade_date >= date('now', ?)
        ORDER BY trade_date DESC
    """
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=(f"-{lookback_days} days",))
    for col in ("filing_date", "trade_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    logger.debug("load_insider_trades: %d rows", len(df))
    return df


def load_daily_prices(
    tickers: list[str] | None = None,
    lookback_days: int = 35,
    db_path: str | Path = _DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Load daily prices for given tickers (all if None) for last N days."""
    if tickers:
        placeholders = ",".join("?" * len(tickers))
        sql = f"""
            SELECT * FROM daily_prices
            WHERE ticker IN ({placeholders})
              AND date >= date('now', ?)
            ORDER BY ticker, date
        """
        params = tickers + [f"-{lookback_days} days"]
    else:
        sql = """
            SELECT * FROM daily_prices
            WHERE date >= date('now', ?)
            ORDER BY ticker, date
        """
        params = [f"-{lookback_days} days"]

    with get_connection(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    logger.debug("load_daily_prices: %d rows", len(df))
    return df


def load_last_scored_setups(
    run_date: str | None = None,
    db_path: str | Path = _DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Load the most recent (or a specific date's) scored setups."""
    if run_date:
        sql = "SELECT * FROM scored_setups WHERE run_date = ? ORDER BY composite_score DESC"
        params: list = [run_date]
    else:
        sql = """
            SELECT * FROM scored_setups
            WHERE run_date = (SELECT MAX(run_date) FROM scored_setups)
            ORDER BY composite_score DESC
        """
        params = []
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    logger.debug("load_last_scored_setups: %d rows", len(df))
    return df


def load_all_scored_setups(db_path: str | Path = _DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Load every row from scored_setups across all run_dates (sans filtre date).

    signal_reasons est retourné tel quel (chaîne pipe-délimitée).

    Returns:
        DataFrame avec toutes les colonnes de scored_setups, trié par
        (run_date, ticker).
    """
    sql = "SELECT * FROM scored_setups ORDER BY run_date, ticker"
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(sql, conn)
    logger.debug("load_all_scored_setups: %d rows across all run_dates", len(df))
    return df


def load_all_daily_prices(
    tickers: list[str],
    db_path: str | Path = _DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    Load toutes les lignes daily_prices pour les tickers donnés, SANS filtre date.

    Utilisé par le moteur de backtest après un fetch yfinance 90j afin d'avoir
    l'historique complet disponible pour les lookups T+5/T+10/T+20.

    Args:
        tickers: liste de tickers (ex. ['VLE.V', 'NVNI.CN'])

    Returns:
        DataFrame avec colonnes [date (datetime64), ticker, close],
        trié par (ticker, date).
    """
    if not tickers:
        return pd.DataFrame(columns=["date", "ticker", "close"])

    placeholders = ",".join("?" * len(tickers))
    sql = f"""
        SELECT date, ticker, close
          FROM daily_prices
         WHERE ticker IN ({placeholders})
         ORDER BY ticker, date
    """
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=tickers)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    logger.debug(
        "load_all_daily_prices: %d rows for %d tickers (no date cutoff)",
        len(df),
        df["ticker"].nunique() if not df.empty else 0,
    )
    return df


def upsert_backtest_results(
    df: pd.DataFrame,
    db_path: str | Path = _DEFAULT_DB_PATH,
) -> int:
    """
    Insert ou remplace des lignes dans backtest_results.

    Utilise INSERT OR REPLACE (clé UNIQUE signal_date + ticker) pour que
    les re-runs du backtest mettent à jour les retours sans créer de doublons.

    Note: les colonnes REAL nullable (close_tN, return_Nd) sont transmises
    comme None (pas 0) afin que SQLite les stocke en NULL.

    Args:
        df: DataFrame avec les colonnes du schéma backtest_results.

    Returns:
        Nombre de lignes écrites.
    """
    if df.empty:
        return 0

    cols = [
        "signal_date", "ticker", "grade", "composite_score",
        "accumulation_flag", "pre_breakout_flag", "cluster_buying_flag",
        "close_at_signal", "close_t5", "close_t10", "close_t20",
        "return_5d", "return_10d", "return_20d",
    ]

    out = df.copy()

    # Colonnes texte et flags
    out["signal_date"] = out["signal_date"].astype(str)
    out["ticker"]      = out["ticker"].astype(str)
    out["grade"]       = out["grade"].fillna("").astype(str)
    for flag_col in ("accumulation_flag", "pre_breakout_flag", "cluster_buying_flag"):
        if flag_col in out.columns:
            out[flag_col] = out[flag_col].fillna(0).astype(int)

    # Colonnes REAL nullable — conserver NaN (sqlite3 les mappe en NULL)
    nullable = [
        "composite_score",
        "close_at_signal", "close_t5", "close_t10", "close_t20",
        "return_5d", "return_10d", "return_20d",
    ]
    for col in nullable:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Construire les tuples en remplaçant NaN par None
    rows = []
    for _, row in out.reindex(columns=cols).iterrows():
        rows.append(tuple(None if pd.isna(v) else v for v in row))

    sql = (
        f"INSERT OR REPLACE INTO backtest_results ({', '.join(cols)}) "
        f"VALUES ({', '.join(['?'] * len(cols))})"
    )
    with get_connection(db_path) as conn:
        conn.executemany(sql, rows)
        conn.commit()

    logger.info("upsert_backtest_results: %d rows", len(rows))
    return len(rows)


def load_backtest_results(db_path: str | Path = _DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Charge tous les résultats de backtest triés par signal_date DESC.
    """
    sql = "SELECT * FROM backtest_results ORDER BY signal_date DESC, ticker"
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(sql, conn)
    logger.debug("load_backtest_results: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _prep(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Select and coerce columns, converting datetimes to ISO strings."""
    out = df.reindex(columns=cols).copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d").fillna("")
        elif out[col].dtype == object:
            out[col] = out[col].fillna("").astype(str)
        else:
            out[col] = out[col].fillna(0)
    return out
