"""
live_bid.py — Live bid validation using a trained model.txt

Usage:
    python live_bid.py --token-id <TOKEN_ID> [--model model.txt] [--threshold 0.6]

What it does:
    1. Fetches recent trades from Polymarket CLOB API (last ~4 hours)
    2. Fetches market metadata from Polymarket Gamma API
    3. Aggregates trades into 5-min buckets (same as training pipeline)
    4. Computes the same features the model was trained on
    5. Predicts P(win) for the most recent complete bucket
    6. Outputs BID or NO BID

The token-id is the 256-bit token ID from markets.csv (token1 = YES, token2 = NO).
Find it in markets.csv or on the Polymarket website URL / API.

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
import math
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import polars as pl
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Polymarket API endpoints
# ---------------------------------------------------------------------------
CLOB_API = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"

# Bucket size (must match training)
BUCKET_MINUTES = 5
WHALE_THRESHOLD_USD = 1000.0

# How many past buckets we need for the longest rolling window (48 buckets)
MIN_HISTORY_BUCKETS = 60  # 60 × 5min = 5 hours, safe margin

# Lag / rolling config (must match config.py FeatureConfig defaults)
LAG_BUCKETS = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [6, 12, 24, 48]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None) -> dict | list:
    """Simple GET with error handling. Requires `requests`."""
    try:
        import requests
    except ImportError:
        print("ERROR: 'requests' is not installed. Run: pip install requests")
        sys.exit(2)

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_trades(token_id: str, limit: int = 1000) -> list[dict]:
    """
    Fetch recent trades for a token from the CLOB API.

    Returns a list of dicts with keys:
        timestamp, price, size (in tokens), side
    """
    data = _get(f"{CLOB_API}/trades", params={
        "token_id": token_id,
        "limit": limit,
    })

    # CLOB returns {"data": [...], "next_cursor": ...} or just a list
    if isinstance(data, dict):
        trades = data.get("data", [])
    else:
        trades = data

    return trades


def fetch_market_info(token_id: str) -> dict:
    """
    Fetch market metadata from Gamma API by clob_token_id.

    Returns a dict with market fields (volume, close_time, etc.)
    or an empty dict if not found.
    """
    try:
        data = _get(f"{GAMMA_API}/markets", params={
            "clob_token_ids": token_id,
        })
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict):
            markets = data.get("markets", data.get("data", []))
            if markets:
                return markets[0]
    except Exception as e:
        print(f"  Warning: could not fetch market metadata: {e}")
    return {}


# ---------------------------------------------------------------------------
# Trade parsing & aggregation
# ---------------------------------------------------------------------------

def parse_trades(raw_trades: list[dict], token_id: str) -> pl.DataFrame:
    """
    Convert raw CLOB API trade dicts into a normalised Polars DataFrame.

    CLOB trade fields (typical):
        type, token_id, outcome, price, size, trader_side,
        timestamp (ISO string or Unix), transaction_hash, ...

    We produce:
        timestamp (Datetime[us]), price (f64), usd_amount (f64),
        token_amount (f64), is_yes (bool)
    """
    rows = []
    for t in raw_trades:
        # Timestamp: may be ISO string or Unix int
        ts_raw = t.get("timestamp") or t.get("created_at") or t.get("time")
        if ts_raw is None:
            continue

        if isinstance(ts_raw, (int, float)):
            ts = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)
        else:
            try:
                ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            except ValueError:
                continue

        price_raw = t.get("price")
        size_raw = t.get("size") or t.get("amount")
        if price_raw is None or size_raw is None:
            continue

        price = float(price_raw)
        token_amount = float(size_raw)
        usd_amount = price * token_amount

        # Determine YES/NO from outcome field or token_id match
        outcome = str(t.get("outcome", "")).upper()
        if outcome in ("YES", "1"):
            is_yes = True
        elif outcome in ("NO", "0"):
            is_yes = False
        else:
            # Fall back: check if this trade's token_id matches our token_id
            is_yes = str(t.get("token_id", "")) == str(token_id)

        rows.append({
            "timestamp": ts.replace(tzinfo=None),  # naive UTC
            "price": price,
            "usd_amount": usd_amount,
            "token_amount": token_amount,
            "is_yes": is_yes,
        })

    if not rows:
        return pl.DataFrame(schema={
            "timestamp": pl.Datetime("us"),
            "price": pl.Float64,
            "usd_amount": pl.Float64,
            "token_amount": pl.Float64,
            "is_yes": pl.Boolean,
        })

    df = pl.DataFrame(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us"))
    )
    return df.sort("timestamp")


def aggregate_to_buckets(trades: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate trades into 5-minute buckets.
    Mirrors pipeline/aggregation.py exactly.
    """
    bucket_dur = f"{BUCKET_MINUTES}m"

    bucketed = (
        trades
        .with_columns(
            pl.col("timestamp").dt.truncate(bucket_dur).alias("bucket_time")
        )
        .group_by("bucket_time")
        .agg([
            pl.len().alias("trade_count"),
            pl.col("usd_amount").sum().alias("total_usd"),
            pl.col("token_amount").sum().alias("total_tokens"),
            pl.col("price").first().alias("open_price"),
            pl.col("price").last().alias("close_price"),
            pl.col("price").max().alias("high_price"),
            pl.col("price").min().alias("low_price"),
            pl.col("price").mean().alias("mean_price"),
            (pl.col("price") * pl.col("usd_amount")).sum().alias("_price_x_usd"),
            pl.col("is_yes").mean().alias("yes_ratio"),
            (pl.col("usd_amount") > WHALE_THRESHOLD_USD).sum().alias("whale_count"),
            pl.col("usd_amount")
                .filter(pl.col("usd_amount") > WHALE_THRESHOLD_USD)
                .sum()
                .alias("whale_usd"),
        ])
        .sort("bucket_time")
        .with_columns(
            (pl.col("_price_x_usd") / pl.col("total_usd"))
                .fill_nan(None)
                .alias("vwap"),
        )
        .drop("_price_x_usd")
        .with_columns(
            (pl.col("close_price") - pl.col("open_price")).alias("momentum"),
            pl.col("whale_usd").fill_null(0.0),
            pl.col("whale_count").fill_null(0),
        )
    )

    return bucketed


# ---------------------------------------------------------------------------
# Feature engineering (mirrors pipeline/features.py)
# ---------------------------------------------------------------------------

def _add_lag_features(df: pl.DataFrame) -> pl.DataFrame:
    lag_cols = ["mean_price", "total_usd", "trade_count", "momentum", "yes_ratio"]
    exprs = []
    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in LAG_BUCKETS:
            exprs.append(pl.col(col).shift(lag).alias(f"{col}_lag{lag}"))

    if exprs:
        df = df.with_columns(exprs)

    for lag in LAG_BUCKETS:
        lag_col = f"mean_price_lag{lag}"
        if lag_col in df.columns:
            df = df.with_columns([
                (pl.col("mean_price") - pl.col(lag_col))
                    .alias(f"price_change_lag{lag}"),
                ((pl.col("mean_price") - pl.col(lag_col)) / pl.col(lag_col))
                    .fill_nan(None)
                    .alias(f"price_return_lag{lag}"),
            ])
    return df


def _add_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    for window in ROLLING_WINDOWS:
        df = df.with_columns([
            pl.col("mean_price").rolling_mean(window_size=window)
                .alias(f"price_ma{window}"),
            pl.col("mean_price").rolling_std(window_size=window)
                .alias(f"price_std{window}"),
            pl.col("total_usd").rolling_sum(window_size=window)
                .alias(f"volume_sum{window}"),
            pl.col("trade_count").rolling_mean(window_size=window)
                .alias(f"trade_count_ma{window}"),
            pl.col("momentum").rolling_mean(window_size=window)
                .alias(f"momentum_ma{window}"),
        ])

    for window in ROLLING_WINDOWS:
        ma_col = f"price_ma{window}"
        df = df.with_columns(
            ((pl.col("mean_price") - pl.col(ma_col)) / pl.col(ma_col))
                .fill_nan(None)
                .alias(f"price_vs_ma{window}")
        )
    return df


def _add_market_features(df: pl.DataFrame, market: dict, now: datetime) -> pl.DataFrame:
    """Add scalar market-level features from Gamma API data."""

    yes_price = _safe_float(market.get("outcomePrices", [None])[0]
                            if market.get("outcomePrices") else market.get("yes_price"))
    no_price = _safe_float(market.get("outcomePrices", [None, None])[1]
                           if market.get("outcomePrices") else market.get("no_price"))
    volume = _safe_float(market.get("volume") or market.get("volumeNum"))
    liquidity = _safe_float(market.get("liquidity") or market.get("liquidityNum"))
    end_date_str = market.get("endDate") or market.get("end_date_iso") or market.get("close_time")

    exprs = []

    if yes_price is not None:
        exprs.append(pl.lit(yes_price).alias("yes_price"))
    if no_price is not None:
        exprs.append(pl.lit(no_price).alias("no_price"))
    if volume is not None:
        exprs.append(pl.lit(volume).alias("volume"))
    if liquidity is not None:
        exprs.append(pl.lit(liquidity).alias("liquidity"))

    if exprs:
        df = df.with_columns(exprs)

    # Derived: spread & entropy
    if "yes_price" in df.columns and "no_price" in df.columns:
        df = df.with_columns(
            (pl.col("yes_price") - pl.col("no_price")).abs().alias("market_spread"),
        )
        p = pl.col("yes_price").clip(1e-9, 1.0 - 1e-9)
        df = df.with_columns(
            (-(p * p.log(base=2) + (1 - p) * (1 - p).log(base=2)))
                .alias("market_entropy")
        )

    # Days to close
    if end_date_str:
        try:
            close_dt = datetime.fromisoformat(
                str(end_date_str).replace("Z", "+00:00")
            ).replace(tzinfo=None)
            days_to_close = (close_dt - now).total_seconds() / 86400
            df = df.with_columns(pl.lit(days_to_close).alias("days_to_close"))
        except Exception:
            pass

    return df


def _add_cross_features(df: pl.DataFrame) -> pl.DataFrame:
    if "yes_price" in df.columns:
        df = df.with_columns(
            (pl.col("mean_price") - pl.col("yes_price")).alias("entry_vs_market")
        )

    if "liquidity" in df.columns:
        df = df.with_columns(
            (pl.col("total_usd") / pl.col("liquidity"))
                .fill_nan(None)
                .alias("trade_size_vs_liquidity")
        )

    if "volume" in df.columns:
        df = df.with_columns(
            (pl.col("total_usd") / pl.col("volume"))
                .fill_nan(None)
                .alias("volume_concentration")
        )

    if "whale_count" in df.columns:
        df = df.with_columns(
            (pl.col("whale_count").cast(pl.Float64) / pl.col("trade_count").cast(pl.Float64))
                .fill_nan(0.0)
                .alias("whale_ratio")
        )
        df = df.with_columns(
            (pl.col("whale_ratio") * pl.col("momentum")).alias("whale_momentum")
        )

    if "days_to_close" in df.columns:
        df = df.with_columns(
            (pl.col("momentum") / (pl.col("days_to_close") + 1))
                .fill_nan(None)
                .alias("momentum_per_day")
        )

    return df


def _add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("bucket_time").dt.hour().alias("hour"),
        pl.col("bucket_time").dt.weekday().alias("day_of_week"),
    ])
    df = df.with_columns([
        (pl.col("hour").cast(pl.Float64) * 2 * math.pi / 24).sin().alias("hour_sin"),
        (pl.col("hour").cast(pl.Float64) * 2 * math.pi / 24).cos().alias("hour_cos"),
        (pl.col("day_of_week").cast(pl.Float64) * 2 * math.pi / 7).sin().alias("dow_sin"),
        (pl.col("day_of_week").cast(pl.Float64) * 2 * math.pi / 7).cos().alias("dow_cos"),
        (pl.col("day_of_week") >= 5).cast(pl.Int8).alias("is_weekend"),
    ])
    return df


def build_features(bucketed: pl.DataFrame, market: dict, now: datetime) -> pl.DataFrame:
    """Build the full feature matrix from bucketed data + market metadata."""
    df = bucketed.clone()
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_market_features(df, market, now)
    df = _add_cross_features(df)
    df = _add_time_features(df)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


EXCLUDE_COLS = {
    "bucket_time", "market_id", "is_empty_bucket", "in_gap", "in_long_gap",
    "exclude_from_training", "win", "future_return", "future_price",
    "question", "close_time",
}


def get_feature_row(df: pl.DataFrame, booster: lgb.Booster) -> np.ndarray:
    """
    Extract the last row of df, aligned to the model's expected feature names.

    Missing features are filled with NaN (LightGBM handles them natively).
    """
    model_features = booster.feature_name()
    last_row = df.tail(1)

    row = {}
    for feat in model_features:
        if feat in last_row.columns:
            val = last_row[feat][0]
            row[feat] = float(val) if val is not None else float("nan")
        else:
            row[feat] = float("nan")

    return np.array([list(row.values())], dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
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
        "--history-buckets", type=int, default=MIN_HISTORY_BUCKETS,
        help="How many past 5-min buckets to fetch (default: 60 = 5 hours).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed feature values.",
    )
    args = parser.parse_args()

    now = datetime.now(tz=timezone.utc).replace(tzinfo=None)

    print(f"\n{'='*55}")
    print(f"  Polymarket Live Bid Validator")
    print(f"{'='*55}")
    print(f"  Token ID : ...{args.token_id[-12:]}")
    print(f"  Model    : {args.model}")
    print(f"  Threshold: P(win) >= {args.threshold:.0%}")
    print(f"  Time     : {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*55}\n")

    # 1. Load model
    print("[1/5] Loading model...")
    try:
        booster = lgb.Booster(model_file=args.model)
    except Exception as e:
        print(f"ERROR: Could not load model from '{args.model}': {e}")
        sys.exit(2)
    print(f"  Loaded. Features: {len(booster.feature_name())}")

    # 2. Fetch market metadata
    print("[2/5] Fetching market metadata...")
    market = fetch_market_info(args.token_id)
    if market:
        q = market.get("question", market.get("title", "unknown"))[:60]
        print(f"  Market : {q}")
        end = market.get("endDate") or market.get("end_date_iso", "unknown")
        print(f"  Closes : {end}")
    else:
        print("  Warning: no market metadata found — some features will be NaN")

    # 3. Fetch recent trades
    print(f"[3/5] Fetching recent trades (last ~{args.history_buckets * BUCKET_MINUTES} min)...")
    raw_trades = fetch_trades(args.token_id, limit=min(args.history_buckets * 20, 2000))
    print(f"  API returned {len(raw_trades)} raw trades")

    trades = parse_trades(raw_trades, args.token_id)

    if trades.is_empty():
        print("ERROR: No trades found. Check the token ID and try again.")
        sys.exit(2)

    # Filter to the lookback window we need
    cutoff = now - timedelta(minutes=args.history_buckets * BUCKET_MINUTES)
    trades = trades.filter(pl.col("timestamp") >= cutoff)
    print(f"  Trades in window: {len(trades)}")

    if len(trades) == 0:
        print("ERROR: No trades in the lookback window. Market may be inactive.")
        sys.exit(2)

    # 4. Aggregate & build features
    print("[4/5] Aggregating and building features...")
    bucketed = aggregate_to_buckets(trades)
    n_buckets = len(bucketed)
    print(f"  Buckets: {n_buckets} × {BUCKET_MINUTES} min")

    if n_buckets < max(LAG_BUCKETS):
        print(f"WARNING: Only {n_buckets} buckets — lag features will be NaN "
              f"(need >{max(LAG_BUCKETS)} for full accuracy).")

    featured = build_features(bucketed, market, now)

    # 5. Predict
    print("[5/5] Predicting...")
    X = get_feature_row(featured, booster)
    p_win = float(booster.predict(X)[0])

    # Decision
    bid = p_win >= args.threshold

    print(f"\n{'='*55}")
    print(f"  P(win)    : {p_win:.4f}  ({p_win:.1%})")
    print(f"  Threshold : {args.threshold:.1%}")
    print(f"  Decision  : {'*** BID ***' if bid else 'NO BID'}")
    print(f"{'='*55}\n")

    if args.verbose:
        model_features = booster.feature_name()
        last = featured.tail(1)
        print("  Feature values (last bucket):")
        for feat in model_features:
            if feat in last.columns:
                v = last[feat][0]
                print(f"    {feat:<35s} = {v}")
            else:
                print(f"    {feat:<35s} = NaN (not computed)")
        print()

    sys.exit(0 if bid else 1)


if __name__ == "__main__":
    main()
