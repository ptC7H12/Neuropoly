"""
Walk-forward time-based train/validation/test split.

Key properties:
- Strictly time-ordered: train < val < test
- Gap and purge are measured in WALL-CLOCK TIME, not in rows.

  Why that matters: the rows of the feature matrix are interleaved across
  thousands of markets, so "12 rows" of gap can be a few seconds of wall
  time while the label looks 30 minutes into the future.  A row-based gap
  therefore lets the last training labels reach into the validation period
  and inflates the out-of-sample scores.  The purge window is derived from
  the label horizon itself so the two can never drift apart.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from config import SplitConfig, LabelConfig


@dataclass
class SplitResult:
    """Container for train/val/test arrays."""

    train_X: np.ndarray
    train_y: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    feature_names: list[str]

    # Realised trade returns, aligned row-for-row with *_y (see labeling.py).
    # None if the labeled data predates the trade_return column.
    train_ret: np.ndarray | None = None
    val_ret: np.ndarray | None = None
    test_ret: np.ndarray | None = None

    # Time boundaries for reference
    train_end: object = None
    val_start: object = None
    val_end: object = None
    test_start: object = None


def purge_minutes(split_cfg: SplitConfig, label_cfg: LabelConfig,
                  bucket_minutes: int) -> int:
    """
    Wall-clock minutes that must separate two splits.

    = the label's own forward window (a row's label depends on prices up to
      that far ahead) + the configured safety gap.
    """
    label_horizon = label_cfg.forward_window_buckets * bucket_minutes
    return label_horizon + split_cfg.split_gap_minutes


def time_split_indices(
    times_us: np.ndarray,
    split_cfg: SplitConfig,
    label_cfg: LabelConfig,
    bucket_minutes: int,
) -> tuple[slice, slice, slice]:
    """
    Compute train/val/test slices for an array of timestamps.

    *times_us* must be sorted ascending, in microseconds since epoch.

    The split ratios pick the two cut *times*; the splits are then trimmed
    back by `purge_minutes` so that no row's forward-looking label can reach
    across a boundary:

        train  [ .......... ]--purge--| val [ ...... ]--purge--| test [ ... ]
                                      ^                        ^
                                 t_train_cut               t_val_cut

    Returns three slices into the (sorted) array.
    """

    n = len(times_us)
    if n == 0:
        raise ValueError("No rows to split.")

    gap_us = purge_minutes(split_cfg, label_cfg, bucket_minutes) * 60 * 1_000_000

    i_train_cut = min(int(n * split_cfg.train_ratio), n - 1)
    i_val_cut = min(
        int(n * (split_cfg.train_ratio + split_cfg.val_ratio)), n - 1
    )

    t_train_cut = int(times_us[i_train_cut])
    t_val_cut = int(times_us[i_val_cut])

    # Trim each split back so its labels stop before the next split starts
    train_end = int(np.searchsorted(times_us, t_train_cut - gap_us, side="left"))
    val_start = int(np.searchsorted(times_us, t_train_cut, side="left"))
    val_end = int(np.searchsorted(times_us, t_val_cut - gap_us, side="left"))
    test_start = int(np.searchsorted(times_us, t_val_cut, side="left"))

    empty = [
        name
        for name, size in (
            ("train", train_end),
            ("val", val_end - val_start),
            ("test", n - test_start),
        )
        if size <= 0
    ]
    if empty:
        span_days = (int(times_us[-1]) - int(times_us[0])) / 86_400_000_000
        raise ValueError(
            f"Walk-forward split produced empty {'/'.join(empty)} set(s). "
            f"The data spans {span_days:.2f} days and each boundary needs "
            f"{purge_minutes(split_cfg, label_cfg, bucket_minutes)} minutes of "
            f"purge. Use a longer history, a shorter --forward-window, or a "
            f"smaller SplitConfig.split_gap_minutes."
        )

    return (
        slice(0, train_end),
        slice(val_start, val_end),
        slice(test_start, n),
    )


def walk_forward_split(
    df: pl.DataFrame,
    feature_cols: list[str],
    split_cfg: SplitConfig,
    label_cfg: LabelConfig,
    bucket_minutes: int = 5,
) -> SplitResult:
    """
    Perform time-based walk-forward split.

    1. Filter to trainable rows (has label, not excluded)
    2. Sort by bucket_time
    3. Cut into train/val/test by time, purging across every boundary
    """

    # Filter to rows that have valid labels and are not excluded
    trainable = df.filter(
        pl.col("win").is_not_null()
        & ~pl.col("exclude_from_training").fill_null(False)
    ).sort("bucket_time")

    n = trainable.height
    if n == 0:
        raise ValueError("No trainable rows after filtering. Check gap config / labels.")

    times_us = trainable["bucket_time"].to_physical().to_numpy()
    tr_sl, va_sl, te_sl = time_split_indices(
        times_us, split_cfg, label_cfg, bucket_minutes
    )

    train_df = trainable[tr_sl]
    val_df = trainable[va_sl]
    test_df = trainable[te_sl]

    # Extract numpy arrays
    existing_features = [c for c in feature_cols if c in trainable.columns]

    def _X(part: pl.DataFrame) -> np.ndarray:
        return part.select(existing_features).to_numpy().astype(np.float32)

    def _y(part: pl.DataFrame) -> np.ndarray:
        return part["win"].to_numpy().astype(np.float32)

    def _ret(part: pl.DataFrame) -> np.ndarray | None:
        if "trade_return" not in part.columns:
            return None
        return part["trade_return"].to_numpy().astype(np.float64)

    return SplitResult(
        train_X=_X(train_df),
        train_y=_y(train_df),
        val_X=_X(val_df),
        val_y=_y(val_df),
        test_X=_X(test_df),
        test_y=_y(test_df),
        feature_names=existing_features,
        train_ret=_ret(train_df),
        val_ret=_ret(val_df),
        test_ret=_ret(test_df),
        train_end=train_df["bucket_time"].max(),
        val_start=val_df["bucket_time"].min(),
        val_end=val_df["bucket_time"].max(),
        test_start=test_df["bucket_time"].min(),
    )


def print_split_info(split: SplitResult) -> None:
    """Print summary of the split."""

    print("\n=== Walk-Forward Split ===")
    print(f"  Train: {split.train_X.shape[0]:>8} rows  |  ends   {split.train_end}")
    print(f"  Val:   {split.val_X.shape[0]:>8} rows  |  {split.val_start} → {split.val_end}")
    print(f"  Test:  {split.test_X.shape[0]:>8} rows  |  starts {split.test_start}")
    print(f"  Features: {split.train_X.shape[1]}")
    print(f"  Train win rate: {split.train_y.mean():.3f}")
    print(f"  Val   win rate: {split.val_y.mean():.3f}")
    print(f"  Test  win rate: {split.test_y.mean():.3f}")
    if split.train_end is not None and split.val_start is not None:
        print(f"  Purge train→val: {split.val_start - split.train_end}")
    if split.val_end is not None and split.test_start is not None:
        print(f"  Purge val→test:  {split.test_start - split.val_end}")
    print("=" * 40)
