# Neuropoly — working notes

LightGBM pipeline predicting P(win) for Polymarket trades. CPU-only.

## Read this first

`docs/universe-and-grouping.md` carries the current line of work: which markets
are trainable, why, and the staged diagnostic (D0-D4) that decides whether
per-group models are worth building. It is written to be picked up cold.

## Where the data comes from

`/root/poly_data` — a fork of `warproxxx/poly_data` **v2**, running on this
same box. **It collects; this repo joins, filters and trains.** It runs with
`PIPELINE_STAGES=collect` (markets + chain), so `processed/trades.csv` is not
produced — see `docs/universe-and-grouping.md` for the measurements behind
that split.

The two-step bridge, in order:

1. `build_registry.py` — `markets.csv` -> `data/market_registry.parquet`.
   Re-run after poly_data appends markets; ids stay stable.
2. `build_trades.py` — `orderFilled.csv` -> `data/trades.parquet`, joined
   against the registry and filtered to the universe *during* the scan. That
   filter is a 16x reduction at event level, not the 3.7x the market counts
   suggest, because the excluded candle instruments are hyperactive.

`convert_to_parquet.py` targets poly_data **v1** and is superseded by both.
Do not repair it; it fails silently (writes an empty `trades.parquet`).

## Environment

```bash
.venv/bin/python -m pytest tests/ -q      # 56 tests, all offline
```

- The venv is `uv`-managed; `uv` lives at `/root/.local/bin/uv`.
- LightGBM needs `libgomp1` (apt). Without it `test_pipeline_e2e` fails with
  `libgomp.so.1: cannot open shared object file`.
- This host is an **LXC guest with no cgroup memory cap and no swap**, so the
  *host* OOM killer takes processes with no trace in `dmesg` or `journalctl` —
  a long run can simply vanish. `/root/poly_data/ram_watchdog.sh` watches
  `MemAvailable` and stops the pipeline before that happens.

## Invariants that break quietly

These have all bitten already. None of them raise.

1. **One row group = one market** (`pipeline/rowgroups.py`). Every lag, rolling
   window and label is `.over("market_id")`. Longest window is 48 buckets =
   4 h, so a market shorter than that yields no complete feature row.
2. **`market_id` must stay unique.** poly_data's `markets.csv` can list a
   market twice, and joining against a duplicated prior registry multiplies
   rows. `build_registry.py` asserts uniqueness before writing.
3. **`market_id` must stay stable** across filter changes and export growth, or
   every Parquet built earlier points at the wrong markets. Ids are therefore
   assigned to *all* markets and reused from the prior registry.
4. **`segments.py` is precision-tuned**, which is correct for *selecting* a
   segment and wrong for *excluding* one. `pipeline/universe.py` widens the
   rules for exclusion; if you touch them, measure **recall**.
5. **`taker_direction`, not `maker_direction`**, maps to Neuropoly's
   `direction`. They are exact inverses; the wrong one flips `is_buy` for the
   whole dataset.
6. **Market identity is deliberately excluded from features**
   (`features.py:403`; `volume` was dropped for the same reason). D3/D4 would
   partially reverse that, which is why the out-of-sample comparison against
   the global baseline is not optional.

## State

- Branch `claude/market-universe-filter`, pushed.
- `data/market_registry.parquet` — all markets, with `keep` / `exclude_reason`.
- `data/trades.parquet` — the universe's events, built by `build_trades.py`.
- **Next: D0** (`census_markets.py`) — the trainability census that decides
  whether the per-group question is worth pursuing at all.

## Conventions

- Tests use **real slugs** from the export, never invented ones — a made-up
  slug proves nothing about a regex written against production naming.
- Every test file runs standalone (`python tests/test_x.py`) and under pytest.
- Docstrings explain *why*, with the measured number that motivated the rule.
