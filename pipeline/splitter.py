"""
Walk-forward time-based train/validation/test split.

Key properties:
- Strictly time-ordered: train < val < test
- Gap between splits to prevent label leakage
- Purge zone removes rows whose forward-looking label
  could overlap into the next split
- Excludes rows in gap periods
"""

import polars as pl
import numpy as np
from dataclasses import dataclass

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

    # Time boundaries for reference
    train_end: object = None
    val_start: object = None
    val_end: object = None
    test_start: object = None


def walk_forward_split(
    df: pl.DataFrame,
    feature_cols: list[str],
    split_cfg: SplitConfig,
    label_cfg: LabelConfig,
) -> SplitResult:
    """
    Perform time-based walk-forward split.

    1. Filter to trainable rows (has label, not excluded)
    2. Sort by bucket_time
    3. Split by time ratios
    4. Insert gaps between splits
    5. Purge forward-looking labels near boundaries
    """

    # Filter to rows that have valid labels and are not excluded
    trainable = df.filter(
        pl.col("win").is_not_null()
        & ~pl.col("exclude_from_training").fill_null(False)
    ).sort("bucket_time")

    n = trainable.height
    if n == 0:
        raise ValueError("No trainable rows after filtering. Check gap config / labels.")

    # Compute split indices
    n_train = int(n * split_cfg.train_ratio)
    n_val = int(n * split_cfg.val_ratio)
    # n_test = rest

    gap = split_cfg.split_gap_buckets
    purge = label_cfg.forward_window_buckets

    # Train: [0, n_train - purge)
    # Gap:   [n_train - purge, n_train + gap)
    # Val:   [n_train + gap, n_train + gap + n_val - purge)
    # Gap:   [n_train + gap + n_val - purge, n_train + gap + n_val + gap)
    # Test:  [n_train + gap + n_val + gap, end)

    train_end = n_train - purge
    val_start = n_train + gap
    val_end = val_start + n_val - purge
    test_start = val_end + gap

    if test_start >= n:
        # Reduce gaps if dataset is too small
        gap = max(1, gap // 2)
        purge = max(1, purge // 2)
        train_end = n_train - purge
        val_start = n_train + gap
        val_end = val_start + n_val - purge
        test_start = val_end + gap

    if test_start >= n:
        raise ValueError(
            f"Dataset too small for split config. "
            f"n={n}, need at least {test_start + 1} rows."
        )

    train_df = trainable.slice(0, max(1, train_end))
    val_df = trainable.slice(val_start, max(1, val_end - val_start))
    test_df = trainable.slice(test_start, n - test_start)

    # Extract numpy arrays
    existing_features = [c for c in feature_cols if c in trainable.columns]

    train_X = train_df.select(existing_features).to_numpy().astype(np.float32)
    train_y = train_df["win"].to_numpy().astype(np.float32)
    val_X = val_df.select(existing_features).to_numpy().astype(np.float32)
    val_y = val_df["win"].to_numpy().astype(np.float32)
    test_X = test_df.select(existing_features).to_numpy().astype(np.float32)
    test_y = test_df["win"].to_numpy().astype(np.float32)

    # Time boundaries
    train_end_time = train_df["bucket_time"].max()
    val_start_time = val_df["bucket_time"].min()
    val_end_time = val_df["bucket_time"].max()
    test_start_time = test_df["bucket_time"].min()

    return SplitResult(
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        test_X=test_X,
        test_y=test_y,
        feature_names=existing_features,
        train_end=train_end_time,
        val_start=val_start_time,
        val_end=val_end_time,
        test_start=test_start_time,
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
    print("=" * 40)
