"""
Unit tests for pipeline/data_loader normalisation.
Run with:  python tests/test_data_loader.py   (or: pytest tests/)
"""

import sys
from datetime import datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DataConfig
from pipeline.data_loader import load_trades


def _write_trades_csv(path: Path) -> None:
    """Two markets, both sides traded, YES quoted at 0.80."""
    pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 0, 0)] * 4,
            "market_id": [1, 1, 2, 2],
            "side": ["token1", "token2", "token1", "token2"],
            # token1 rows quote the YES price, token2 rows the NO price
            "price": [0.80, 0.20, 0.35, 0.65],
            "usd_amount": [100.0, 100.0, 50.0, 50.0],
            "token_amount": [125.0, 500.0, 142.85, 76.9],
        }
    ).write_csv(path)


def test_no_side_prices_are_normalised_to_yes():
    """
    A NO share at 0.20 is the same market state as a YES share at 0.80.
    Averaging them raw would give mean_price 0.50 and make the price column
    move with the YES/NO trade mix instead of the market.
    """
    tmp = Path("_test_trades.csv")
    _write_trades_csv(tmp)
    try:
        cfg = DataConfig(trades_path=str(tmp), trades_format="csv")
        df = load_trades(cfg).collect()

        m1 = df.filter(pl.col("market_id") == 1)
        assert m1["price"].to_list() == [0.80, 0.80], (
            f"NO price not flipped to YES: {m1['price'].to_list()}"
        )
        assert abs(m1["price"].mean() - 0.80) < 1e-9, (
            "mean price should be the market price, not a side blend"
        )

        m2 = df.filter(pl.col("market_id") == 2)
        assert m2["price"].to_list() == [0.35, 0.35], m2["price"].to_list()

        # is_yes must still record the side mix — that is a separate signal
        assert df["is_yes"].to_list() == [1, 0, 1, 0]

        print("  price normalisation:            OK")
    finally:
        tmp.unlink(missing_ok=True)


def test_unmapped_side_is_left_untouched():
    """A side value we cannot map (is_yes = -1) must not be flipped."""
    tmp = Path("_test_trades2.csv")
    pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1)],
            "market_id": [1],
            "side": ["something_else"],
            "price": [0.30],
            "usd_amount": [10.0],
            "token_amount": [33.0],
        }
    ).write_csv(tmp)
    try:
        cfg = DataConfig(trades_path=str(tmp), trades_format="csv")
        df = load_trades(cfg).collect()
        assert df["is_yes"][0] == -1
        assert df["price"][0] == 0.30, "unmapped side must not be flipped"
        print("  unmapped side untouched:        OK")
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    print("=" * 60)
    print("  data_loader unit tests")
    print("=" * 60)
    test_no_side_prices_are_normalised_to_yes()
    test_unmapped_side_is_left_untouched()
    print("  ALL PASSED")
