#!/usr/bin/env python3
"""
collect_trades.py — Continuous trade collector daemon for Polymarket.

Polls the PUBLIC data-api every POLL_INTERVAL seconds and writes new trades
to a local SQLite database.  live_bid.py and paper_trades.py can then read
from this database instead of hitting the API, which gives them reliable
history even for low-activity markets.

Why per MARKET and not per token
--------------------------------
The training pipeline aggregates both tokens of a market into one bucket, so
`yes_ratio` means "share of fills on the YES side".  Collecting a single
token would pin that feature to a constant and break the very signal the
label is built on.  You still pass a token ID on the command line — it is
resolved to the market's conditionId via the Gamma API.

Usage:
    python collect_trades.py \
        --token-ids <TOKEN_A> <TOKEN_B> \
        --db trades.db \
        --poll-interval 60

    python live_bid.py --token-id <TOKEN_A> --db trades.db --model model.txt

Schema (table `trades`):
    trade_key    TEXT  PK  — deduplication key
    condition_id TEXT      — market (both tokens land here)
    asset        TEXT      — CLOB token ID of this fill
    timestamp    INTEGER   — Unix timestamp (seconds UTC)
    price        REAL      — price as traded (of `asset`)
    price_yes    REAL      — same trade expressed as P(YES)
    usd_amount   REAL      — trade value in USD
    token_amount REAL      — token quantity
    is_yes       INTEGER   — 1 = YES side, 0 = NO side
    tx_hash      TEXT      — on-chain transaction hash
"""

import argparse
import hashlib
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.polymarket_api import (
    PolymarketAPIError,
    MarketInfo,
    fetch_market,
    fetch_trades,
)

_SCHEMA_COLUMNS = {
    "trade_key", "condition_id", "asset", "timestamp",
    "price", "price_yes", "usd_amount", "token_amount", "is_yes", "tx_hash",
}


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure the schema exists."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads while writing
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_key     TEXT    PRIMARY KEY,
            condition_id  TEXT    NOT NULL,
            asset         TEXT    NOT NULL,
            timestamp     INTEGER NOT NULL,
            price         REAL    NOT NULL,
            price_yes     REAL    NOT NULL,
            usd_amount    REAL    NOT NULL,
            token_amount  REAL    NOT NULL,
            is_yes        INTEGER NOT NULL,
            tx_hash       TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_trades_market_ts
        ON trades (condition_id, timestamp DESC)
    """)
    conn.commit()
    _check_schema(conn, db_path)
    return conn


def _check_schema(conn: sqlite3.Connection, db_path: str) -> None:
    """
    Refuse to run against a database written by the pre-data-api version.

    That schema was keyed on token_id and its is_yes column was always 1
    (it compared each trade's token against the one token being queried),
    so mixing the two would silently poison yes_ratio.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
    missing = _SCHEMA_COLUMNS - cols
    if missing:
        conn.close()
        print(
            f"ERROR: '{db_path}' uses an old schema (missing: "
            f"{', '.join(sorted(missing))}).\n"
            f"       That data was collected per token and its is_yes column "
            f"is unusable.\n"
            f"       Delete the file and start a fresh collection."
        )
        sys.exit(1)


def insert_trades(conn: sqlite3.Connection, rows: list[tuple]) -> int:
    """Insert new trade rows, ignoring duplicates. Returns rows inserted."""
    if not rows:
        return 0
    cur = conn.executemany(
        """INSERT OR IGNORE INTO trades
           (trade_key, condition_id, asset, timestamp, price, price_yes,
            usd_amount, token_amount, is_yes, tx_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return cur.rowcount


def prune_old_trades(conn: sqlite3.Connection, keep_seconds: int = 86400 * 7) -> None:
    """Delete trades older than keep_seconds (default: 7 days)."""
    cutoff = int(time.time()) - keep_seconds
    conn.execute("DELETE FROM trades WHERE timestamp < ?", (cutoff,))
    conn.commit()


def latest_timestamp(conn: sqlite3.Connection, condition_id: str) -> int:
    """Unix timestamp of the most recent stored trade for a market (or 0)."""
    row = conn.execute(
        "SELECT MAX(timestamp) FROM trades WHERE condition_id = ?", (condition_id,)
    ).fetchone()
    return row[0] or 0


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _trade_key(trade: dict) -> str:
    """
    Stable deduplication key.

    One transaction can contain several fills, so tx_hash alone would drop
    real trades.  Hash the fields that together identify a fill.
    """
    parts = "|".join(
        str(trade.get(k, ""))
        for k in ("transactionHash", "asset", "timestamp", "price", "size",
                  "side", "proxyWallet")
    )
    return hashlib.sha1(parts.encode("utf-8")).hexdigest()


def fetch_and_store(conn: sqlite3.Connection, market: MarketInfo,
                    lookback_seconds: int) -> int:
    """
    Fetch recent trades for a market and store them.  Returns rows inserted.

    Re-fetches a small overlap window rather than only strictly newer trades:
    the API can report a fill slightly out of order, and INSERT OR IGNORE
    makes re-reading it free.
    """
    from pipeline.polymarket_api import _resolve_side  # side resolution rules

    last_ts = latest_timestamp(conn, market.condition_id)
    since = max(0, (last_ts or int(time.time()) - lookback_seconds) - 300)

    raw = fetch_trades(market.condition_id, since_unix=since, max_trades=5000)

    rows = []
    for t in raw:
        ts = t.get("timestamp")
        price = t.get("price")
        size = t.get("size")
        if ts is None or price is None or size is None:
            continue
        is_yes = _resolve_side(t, market)
        if is_yes is None:
            # Never guess: a wrong side silently corrupts yes_ratio
            continue

        price = float(price)
        size = float(size)
        rows.append((
            _trade_key(t),
            market.condition_id,
            str(t.get("asset", "")),
            int(ts),
            price,
            price if is_yes == 1 else 1.0 - price,
            price * size,
            size,
            is_yes,
            str(t.get("transactionHash") or ""),
        ))

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
        help="One or more Polymarket token IDs. Each is resolved to its "
             "market, and BOTH tokens of that market are collected.",
    )
    parser.add_argument(
        "--db", default="trades.db",
        help="Path to SQLite database (default: trades.db).",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=60,
        help="Seconds between API polls (default: 60).",
    )
    parser.add_argument(
        "--keep-days", type=int, default=7,
        help="Days of history to keep in the DB (default: 7).",
    )
    parser.add_argument(
        "--backfill-hours", type=int, default=12,
        help="How much history to pull on the first poll (default: 12).",
    )
    args = parser.parse_args()

    conn = init_db(args.db)

    # Resolve every token to its market once, up front
    print(f"\nResolving {len(args.token_ids)} token(s) …")
    markets: dict[str, MarketInfo] = {}
    for tid in args.token_ids:
        try:
            info = fetch_market(tid)
        except PolymarketAPIError as exc:
            print(f"  ...{tid[-12:]}  ERROR: {exc}")
            continue
        if info is None or not info.condition_id:
            print(f"  ...{tid[-12:]}  not found — skipping")
            continue
        markets[info.condition_id] = info
        print(f"  ...{tid[-12:]}  → {info.question[:52]}")

    if not markets:
        print("No markets could be resolved. Check the token IDs.")
        sys.exit(1)

    # Graceful shutdown on Ctrl+C / SIGTERM
    running = True

    def _stop(sig, frame):
        nonlocal running
        print("\nShutting down collector...")
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"\nCollector started  —  DB: {args.db}")
    print(f"Tracking {len(markets)} market(s) every {args.poll_interval}s")
    print("Press Ctrl+C to stop.\n")

    prune_interval = 3600  # prune once per hour
    last_prune = time.time()

    while running:
        now_str = datetime.now(tz=timezone.utc).strftime("%H:%M:%S UTC")
        for market in markets.values():
            try:
                n_new = fetch_and_store(
                    conn, market, lookback_seconds=args.backfill_hours * 3600
                )
                print(f"[{now_str}] {market.question[:40]:<40}  +{n_new} new trades")
            except (PolymarketAPIError, sqlite3.Error) as exc:
                print(f"[{now_str}] {market.question[:40]:<40}  ERROR: {exc}")

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
