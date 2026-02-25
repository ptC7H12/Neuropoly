"""
collect_trades.py — Continuous trade collector daemon for Polymarket.

Polls the Polymarket CLOB API every POLL_INTERVAL seconds and writes
new trades to a local SQLite database.  live_bid.py can then read from
this database instead of the API, giving it accurate 4h+ history even
for low-activity markets.

Usage:
    # Start the collector for one or more token IDs
    python collect_trades.py \
        --token-ids <TOKEN_A> <TOKEN_B> \
        --db trades.db \
        --poll-interval 60

    # live_bid.py then reads from that DB:
    python live_bid.py \
        --token-id <TOKEN_A> \
        --db trades.db \
        --model model.txt

Schema (table `trades`):
    token_id    TEXT      — Polymarket 256-bit token ID
    timestamp   INTEGER   — Unix timestamp (seconds UTC)
    price       REAL      — Trade price (0–1)
    usd_amount  REAL      — Trade value in USD
    token_amount REAL     — Token quantity
    is_yes      INTEGER   — 1 = YES trade, 0 = NO trade
    tx_hash     TEXT      — On-chain transaction hash (deduplication key)
"""

import argparse
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Polymarket CLOB API
# ---------------------------------------------------------------------------
CLOB_API = "https://clob.polymarket.com"


def _get(url: str, params: dict | None = None) -> list | dict:
    try:
        import requests
    except ImportError:
        print("ERROR: pip install requests")
        sys.exit(1)
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure the schema exists."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads while writing
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            token_id      TEXT    NOT NULL,
            timestamp     INTEGER NOT NULL,
            price         REAL    NOT NULL,
            usd_amount    REAL    NOT NULL,
            token_amount  REAL    NOT NULL,
            is_yes        INTEGER NOT NULL,
            tx_hash       TEXT    NOT NULL,
            PRIMARY KEY (token_id, tx_hash)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_trades_token_ts
        ON trades (token_id, timestamp DESC)
    """)
    conn.commit()
    return conn


def latest_timestamp(conn: sqlite3.Connection, token_id: str) -> int:
    """Return the Unix timestamp of the most recent stored trade (or 0)."""
    row = conn.execute(
        "SELECT MAX(timestamp) FROM trades WHERE token_id = ?", (token_id,)
    ).fetchone()
    return row[0] or 0


def insert_trades(conn: sqlite3.Connection, rows: list[tuple]) -> int:
    """
    Insert new trade rows, ignoring duplicates (by tx_hash).
    Returns the number of newly inserted rows.
    """
    if not rows:
        return 0
    cur = conn.executemany(
        """INSERT OR IGNORE INTO trades
           (token_id, timestamp, price, usd_amount, token_amount, is_yes, tx_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return cur.rowcount


def prune_old_trades(conn: sqlite3.Connection, keep_seconds: int = 86400 * 7) -> None:
    """Delete trades older than keep_seconds (default: 7 days)."""
    cutoff = int(time.time()) - keep_seconds
    conn.execute("DELETE FROM trades WHERE timestamp < ?", (cutoff,))
    conn.commit()


# ---------------------------------------------------------------------------
# Trade fetching & parsing
# ---------------------------------------------------------------------------

def fetch_and_store(conn: sqlite3.Connection, token_id: str) -> int:
    """
    Fetch new trades for token_id from the CLOB API and store them in SQLite.
    Only fetches trades newer than the latest stored timestamp.
    Returns number of new trades inserted.
    """
    last_ts = latest_timestamp(conn, token_id)

    raw = _get(f"{CLOB_API}/trades", params={"token_id": token_id, "limit": 500})
    if isinstance(raw, dict):
        raw = raw.get("data", [])

    rows = []
    for t in raw:
        ts_raw = t.get("timestamp") or t.get("created_at") or t.get("time")
        if ts_raw is None:
            continue

        if isinstance(ts_raw, (int, float)):
            ts = int(float(ts_raw))
        else:
            try:
                ts = int(datetime.fromisoformat(
                    str(ts_raw).replace("Z", "+00:00")
                ).timestamp())
            except ValueError:
                continue

        if ts <= last_ts:
            continue  # already stored

        price_raw = t.get("price")
        size_raw = t.get("size") or t.get("amount")
        if price_raw is None or size_raw is None:
            continue

        price = float(price_raw)
        token_amount = float(size_raw)
        usd_amount = price * token_amount
        tx_hash = str(t.get("transaction_hash") or t.get("transactionHash") or
                      t.get("hash") or f"{token_id}_{ts}_{price}")

        outcome = str(t.get("outcome", "")).upper()
        if outcome in ("YES", "1"):
            is_yes = 1
        elif outcome in ("NO", "0"):
            is_yes = 0
        else:
            is_yes = 1 if str(t.get("token_id", "")) == str(token_id) else 0

        rows.append((token_id, ts, price, usd_amount, token_amount, is_yes, tx_hash))

    return insert_trades(conn, rows)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous Polymarket trade collector → SQLite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--token-ids", nargs="+", required=True,
        help="One or more Polymarket token IDs to track.",
    )
    parser.add_argument(
        "--db", default="trades.db",
        help="Path to SQLite database (default: trades.db).",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=60,
        help="Seconds between API polls per token (default: 60).",
    )
    parser.add_argument(
        "--keep-days", type=int, default=7,
        help="Days of history to keep in the DB (default: 7).",
    )
    args = parser.parse_args()

    conn = init_db(args.db)

    # Graceful shutdown on Ctrl+C / SIGTERM
    running = True
    def _stop(sig, frame):
        nonlocal running
        print("\nShutting down collector...")
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"\nCollector started  —  DB: {args.db}")
    print(f"Tracking {len(args.token_ids)} token(s) every {args.poll_interval}s")
    for tid in args.token_ids:
        print(f"  ...{tid[-12:]}")
    print("Press Ctrl+C to stop.\n")

    prune_interval = 3600  # prune once per hour
    last_prune = time.time()

    while running:
        now_str = datetime.now(tz=timezone.utc).strftime("%H:%M:%S UTC")
        for token_id in args.token_ids:
            try:
                n_new = fetch_and_store(conn, token_id)
                print(f"[{now_str}] ...{token_id[-12:]}  +{n_new} new trades")
            except Exception as e:
                print(f"[{now_str}] ...{token_id[-12:]}  ERROR: {e}")

        # Prune old data periodically
        if time.time() - last_prune > prune_interval:
            prune_old_trades(conn, keep_seconds=args.keep_days * 86400)
            last_prune = time.time()

        # Sleep in short increments so Ctrl+C is responsive
        for _ in range(args.poll_interval):
            if not running:
                break
            time.sleep(1)

    conn.close()
    print("Collector stopped.")


if __name__ == "__main__":
    main()
