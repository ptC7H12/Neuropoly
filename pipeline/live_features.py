"""
Feature construction for the live path.

This module deliberately contains almost no logic of its own: it wires the
live inputs into the SAME functions the training pipeline runs, so a feature
computed at inference time is computed by the same code that produced the
training data.

live_bid.py and paper_trades.py used to carry their own copies of the
bucketing, lag, rolling and time features, with the window sizes hardcoded.
Three consequences, all silent:

  * training with --low-memory or --bucket-minutes 15 produced a model whose
    features no longer matched what the live scripts computed;
  * the live copies never filled empty buckets, so `price_ma48` covered
    "the last 48 buckets that happened to have trades" (possibly days)
    instead of the last 4 hours;
  * the live copies predicted on the CURRENT, still-open bucket, while every
    training row is a completed bucket.

All three are handled here.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from config import PipelineConfig
from pipeline.aggregation import aggregate_trades_eager
from pipeline.features import build_features
from pipeline.gap_handler import fill_market_buckets
from pipeline.polymarket_api import MarketInfo

# Synthetic market id for the single market we score live
LIVE_MARKET_ID = 0


def market_frame(market: MarketInfo | None) -> pl.DataFrame:
    """
    One-row markets table in the shape pipeline/features expects.

    NOTE on days_to_close: training reads `closedTime` from markets.csv (when
    the market actually closed), the live path uses Gamma's `endDate` (when it
    is scheduled to close).  For markets that resolve early the two differ.
    """
    schema = {
        "market_id": pl.Int64,
        "yes_price": pl.Float64,
        "no_price": pl.Float64,
        "volume": pl.Float64,
        "liquidity": pl.Float64,
        "close_time": pl.Datetime("us"),
    }
    row = {
        "market_id": LIVE_MARKET_ID,
        "yes_price": market.yes_price if market else None,
        "no_price": market.no_price if market else None,
        "volume": market.volume if market else None,
        "liquidity": market.liquidity if market else None,
        "close_time": market.end_date if market else None,
    }
    return pl.DataFrame([row], schema=schema)


def build_live_features(
    trades: pl.DataFrame,
    market: MarketInfo | None,
    cfg: PipelineConfig,
    now: datetime | None = None,
    drop_open_bucket: bool = True,
) -> pl.DataFrame:
    """
    trades → buckets → gap fill → features, using the training code paths.

    `trades` must already be normalised (see polymarket_api.normalize_trades):
    timestamp / price (as P(YES)) / usd_amount / token_amount / is_yes.

    With drop_open_bucket=True the still-running bucket is removed, so the
    row handed to the model is a completed bucket — the same thing every
    training row is.  Returns an empty frame if nothing is left.
    """
    if trades.is_empty():
        return pl.DataFrame()

    bucketed = aggregate_trades_eager(trades, cfg.bucket, market_id=LIVE_MARKET_ID)
    if bucketed.is_empty():
        return pl.DataFrame()

    if drop_open_bucket:
        now = now or datetime.utcnow()
        current_bucket = _truncate(now, cfg.bucket.bucket_minutes)
        bucketed = bucketed.filter(pl.col("bucket_time") < current_bucket)
        if bucketed.is_empty():
            return pl.DataFrame()

    # Same gap fill as training, so rolling windows span the same wall time
    filled = fill_market_buckets(bucketed, cfg.bucket, cfg.gap)

    # build_features expects these flags to exist (they are excluded from the
    # feature set itself, but gap-aware code paths read them).
    for col in ("in_long_gap", "exclude_from_training"):
        if col not in filled.columns:
            filled = filled.with_columns(pl.lit(False).alias(col))

    return build_features(filled, market_frame(market), cfg.features)


def _truncate(dt: datetime, bucket_minutes: int) -> datetime:
    """
    Floor a timestamp to its bucket boundary.

    Delegates to the same Polars truncation that builds the buckets in
    aggregation.py.  Flooring the minute field by hand only agrees with it
    when bucket_minutes divides 60: Polars truncates from the Unix epoch, so
    at 7 min it lands on 13:47 where minute-flooring gives 13:42, and at
    120 min on 12:00 where minute-flooring gives 13:00.
    """
    return pl.Series([dt]).dt.truncate(f"{bucket_minutes}m")[0]


def bucket_age(last_bucket_time, now: datetime, bucket_minutes: int) -> timedelta:
    """
    How old the scored bucket is, measured from the END of that bucket.

    The live path scores the last bucket that actually EXISTS, which on a
    quiet market can be hours old — in a real test run the scored bucket was
    11:00 while the clock said 13:15.  Every training row, by contrast, is a
    bucket that had just closed.  Scoring a stale bucket is not wrong, but
    calling it a live signal without saying so is.
    """
    bucket_end = last_bucket_time + timedelta(minutes=bucket_minutes)
    return now - bucket_end


def is_stale(age: timedelta, bucket_minutes: int, max_buckets: int = 3) -> bool:
    """True when the scored bucket is older than `max_buckets` bucket widths."""
    return age > timedelta(minutes=max_buckets * bucket_minutes)


def history_window(cfg: PipelineConfig, history_buckets: int) -> timedelta:
    """How far back the live path has to look to fill every rolling window."""
    needed = max(
        history_buckets,
        max(cfg.features.rolling_windows, default=1),
        max(cfg.features.lag_buckets, default=1),
    )
    return timedelta(minutes=needed * cfg.bucket.bucket_minutes)


def align_to_model(df: pl.DataFrame, model_features: list[str]):
    """
    Last row of `df`, ordered to match the model's own feature names.

    Missing features become NaN, which LightGBM handles natively.  Returns
    (X, missing_feature_names) so callers can warn about a train/serve gap
    instead of silently scoring on NaNs.
    """
    import numpy as np

    last_row = df.tail(1)
    values = []
    missing = []
    for feat in model_features:
        if feat in last_row.columns:
            val = last_row[feat][0]
            values.append(float(val) if val is not None else float("nan"))
        else:
            values.append(float("nan"))
            missing.append(feat)

    return np.array([values], dtype=np.float32), missing


# Features the live path structurally cannot produce, with the reason.
# Kept separate from "your config does not match training" so the warning
# points at the real cause.
UNAVAILABLE_LIVE_FEATURES = {
    "buy_ratio": (
        "derived from orderFilled's maker-side `direction` column, which has "
        "no equivalent in the public trade feed (data-api reports the TAKER "
        "side). Feeding the taker side here would be a different quantity, so "
        "it is left as NaN rather than faked."
    ),
}


def explain_missing(missing: list[str]) -> tuple[list[str], list[str]]:
    """Split missing features into (known-unavailable, unexpected)."""
    known = [f for f in missing if f in UNAVAILABLE_LIVE_FEATURES]
    unexpected = [f for f in missing if f not in UNAVAILABLE_LIVE_FEATURES]
    return known, unexpected
