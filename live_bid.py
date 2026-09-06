#!/usr/bin/env python3
"""
live_bid.py — Live bid validation using a trained model.txt

Usage:
    python live_bid.py --token-id <TOKEN_ID> [--model model.txt] [--threshold 0.6]

What it does:
    1. Resolves the token to its market via the Polymarket Gamma API
    2. Fetches recent trades for that MARKET (both tokens) from the public
       data-api, or from the SQLite DB written by collect_trades.py
    3. Buckets, gap-fills and builds features with the SAME pipeline code
       that produced the training data (pipeline/live_features.py)
    4. Predicts P(win) for the most recent COMPLETED bucket
    5. Outputs BID or NO BID

The token-id is the 256-bit token ID from markets.csv (token1 = YES, token2 =
NO).  Find it in markets.csv or on the Polymarket website URL / API.

Feature parity with training matters: pass the same --bucket-minutes and
--low-memory you trained with, otherwise the features will not line up and
the model is scoring something else.

Example:
    python live_bid.py \
        --token-id 21742633143463906290569050155826241533067272736897614950488156847949938836455 \
        --model model.txt \
        --threshold 0.6

Exit codes:
    0  — BID (P(win) >= threshold)
    1  — NO BID (P(win) < threshold)
    2  — Error (not enough data, API failure, etc.)
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig
from pipeline.live_features import (
    UNAVAILABLE_LIVE_FEATURES,
    align_to_model,
    build_live_features,
    explain_missing,
    history_window,
)
from pipeline.polymarket_api import (
    MarketInfo,
    PolymarketAPIError,
    fetch_market,
    fetch_trades,
    normalize_trades,
)

_TRADE_SCHEMA = {
    "timestamp": pl.Datetime("us"),
    "price": pl.Float64,
    "usd_amount": pl.Float64,
    "token_amount": pl.Float64,
    "is_yes": pl.Int8,
}


def fetch_trades_from_db(
    db_path: str,
    condition_id: str,
    since_unix: int,
) -> pl.DataFrame:
    """
    Read recent trades for a market from the collect_trades.py database.

    `price_yes` is used, so the frame carries P(YES) exactly like the batch
    pipeline does after data_loader normalisation.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
        if "price_yes" not in cols or "condition_id" not in cols:
            raise RuntimeError(
                f"'{db_path}' uses the old per-token schema, whose is_yes "
                f"column is always 1. Delete it and re-collect with the "
                f"current collect_trades.py."
            )
        rows = conn.execute(
            """SELECT timestamp, price_yes, usd_amount, token_amount, is_yes
               FROM trades
               WHERE condition_id = ? AND timestamp >= ?
               ORDER BY timestamp ASC""",
            (condition_id, since_unix),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return pl.DataFrame(schema=_TRADE_SCHEMA)

    return pl.DataFrame(
        {
            "timestamp": [
                datetime.fromtimestamp(r[0], tz=timezone.utc).replace(tzinfo=None)
                for r in rows
            ],
            "price": [float(r[1]) for r in rows],
            "usd_amount": [float(r[2]) for r in rows],
            "token_amount": [float(r[3]) for r in rows],
            "is_yes": [int(r[4]) for r in rows],
        },
        schema=_TRADE_SCHEMA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live bid validation using a trained Polymarket model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--token-id", required=True,
        help="Polymarket token ID (256-bit, from markets.csv token1/token2).",
    )
    parser.add_argument(
        "--model", default="model.txt",
        help="Path to the trained LightGBM model file (default: model.txt).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.6,
        help="P(win) threshold to trigger a BID (default: 0.6).",
    )
    parser.add_argument(
        "--history-buckets", type=int, default=60,
        help="How many past buckets to use (default: 60 = 5 h at 5 min).",
    )
    parser.add_argument(
        "--bucket-minutes", type=int, default=5,
        help="Bucket size — MUST match what the model was trained with.",
    )
    parser.add_argument(
        "--low-memory", action="store_true",
        help="Reduced feature set — set this if you trained with --low-memory.",
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to SQLite DB from collect_trades.py. "
             "If given, reads history from the DB instead of the live API.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed feature values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = PipelineConfig()
    cfg.bucket.bucket_minutes = args.bucket_minutes
    if args.low_memory:
        cfg.features.lag_buckets = [1, 3, 6]
        cfg.features.rolling_windows = [6, 12]
        cfg.features.cross_market_features = False

    now = datetime.now(tz=timezone.utc).replace(tzinfo=None)

    print(f"\n{'='*55}")
    print(f"  Polymarket Live Bid Validator")
    print(f"{'='*55}")
    print(f"  Token ID : ...{args.token_id[-12:]}")
    print(f"  Model    : {args.model}")
    print(f"  Threshold: P(win) >= {args.threshold:.0%}")
    print(f"  Buckets  : {args.bucket_minutes} min")
    print(f"  Time     : {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*55}\n")

    # 1. Load model
    print("[1/5] Loading model...")
    try:
        booster = lgb.Booster(model_file=args.model)
    except Exception as e:
        print(f"ERROR: Could not load model from '{args.model}': {e}")
        sys.exit(2)
    model_features = booster.feature_name()
    print(f"  Loaded. Features: {len(model_features)}")

    # 2. Resolve the market
    print("[2/5] Resolving market...")
    try:
        market: MarketInfo | None = fetch_market(args.token_id)
    except PolymarketAPIError as exc:
        print(f"ERROR: {exc}")
        sys.exit(2)
    if market is None or not market.condition_id:
        print("ERROR: No market found for that token ID.")
        sys.exit(2)
    print(f"  Market : {market.question[:60]}")
    print(f"  Closes : {market.end_date}")
    side_label = {1: "YES", 0: "NO"}.get(market.side_of(args.token_id), "?")
    print(f"  Your token is the {side_label} side")

    # 3. Load recent trades (from SQLite DB or the public data-api)
    lookback = history_window(cfg, args.history_buckets)
    cutoff = now - lookback
    cutoff_unix = int(cutoff.replace(tzinfo=timezone.utc).timestamp())

    if args.db:
        print(f"[3/5] Reading trades from DB ({args.db})...")
        try:
            trades = fetch_trades_from_db(args.db, market.condition_id, cutoff_unix)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            sys.exit(2)
    else:
        print(f"[3/5] Fetching trades from data-api "
              f"(last {lookback.total_seconds()/3600:.1f} h)...")
        try:
            raw_trades = fetch_trades(
                market.condition_id, since_unix=cutoff_unix, max_trades=5000
            )
        except PolymarketAPIError as exc:
            print(f"ERROR: {exc}")
            sys.exit(2)
        print(f"  API returned {len(raw_trades)} trades in window")
        trades = normalize_trades(raw_trades, market)

    print(f"  Usable trades: {trades.height}")
    if trades.is_empty():
        print("ERROR: No trades in the lookback window. Market may be inactive.")
        print("       Use --db with collect_trades.py for low-activity markets.")
        sys.exit(2)

    # 4. Bucket + gap-fill + features, via the training code path
    print("[4/5] Building features (training pipeline)...")
    featured = build_live_features(trades, market, cfg, now=now)
    if featured.is_empty():
        print("ERROR: No completed bucket available yet — wait for the current "
              "bucket to close, or widen --history-buckets.")
        sys.exit(2)

    n_buckets = featured.height
    last_bucket_time = featured["bucket_time"][-1]
    print(f"  Buckets: {n_buckets} × {cfg.bucket.bucket_minutes} min "
          f"(gap-filled, current bucket excluded)")
    print(f"  Scoring bucket: {last_bucket_time}")

    longest = max(cfg.features.rolling_windows, default=1)
    if n_buckets < longest:
        print(f"  WARNING: only {n_buckets} buckets — rolling windows up to "
              f"{longest} will be NaN. Collect more history for full accuracy.")

    # 5. Predict
    print("[5/5] Predicting...")
    X, missing = align_to_model(featured, model_features)
    known, unexpected = explain_missing(missing)
    for feat in known:
        print(f"  NOTE: '{feat}' is NaN — {UNAVAILABLE_LIVE_FEATURES[feat]}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} model feature(s) not produced live "
              f"(scored as NaN): {unexpected[:5]}"
              f"{'...' if len(unexpected) > 5 else ''}")
        print(f"           Check that --bucket-minutes / --low-memory match "
              f"what the model was trained with.")
    p_win = float(booster.predict(X)[0])

    # --- Which side does P(win) refer to? ---
    #
    # During training (labeling.py):
    #   yes_ratio > 0.5  →  bucket is predominantly YES buys
    #                        win=1 means P(YES) went UP   → good for YES holders
    #   yes_ratio <= 0.5 →  bucket is predominantly NO buys
    #                        win=1 means P(YES) went DOWN → good for NO holders
    #
    # So P(win) = probability that the DOMINANT side of that bucket wins.
    last_bucket = featured.tail(1)
    yes_ratio_val = last_bucket["yes_ratio"][0]
    if yes_ratio_val is None:
        yes_ratio_val = 0.5

    dominant_side = "YES" if yes_ratio_val > 0.5 else "NO"
    bid = p_win >= args.threshold

    print(f"\n{'='*55}")
    print(f"  Scored bucket            : {last_bucket_time}")
    print(f"  Bucket yes_ratio         : {yes_ratio_val:.2f}")
    print(f"  Mean price (P(YES))      : {last_bucket['mean_price'][0]:.4f}")
    print(f"  Dominant side            : {dominant_side} "
          f"({'price must rise' if dominant_side == 'YES' else 'price must fall'} to win)")
    print(f"  P(win) for {dominant_side + ' ':<14s}: {p_win:.4f}  ({p_win:.1%})")
    print(f"  Threshold                : {args.threshold:.1%}")
    if bid:
        print(f"  Decision  : *** BID {dominant_side} ***")
    else:
        print(f"  Decision  : NO BID  (P(win) too low)")
    print(f"{'='*55}\n")

    if args.verbose:
        last = featured.tail(1)
        print("  Feature values (scored bucket):")
        for feat in model_features:
            if feat in last.columns:
                print(f"    {feat:<35s} = {last[feat][0]}")
            else:
                print(f"    {feat:<35s} = NaN (not computed)")
        print()

    sys.exit(0 if bid else 1)


if __name__ == "__main__":
    main()
