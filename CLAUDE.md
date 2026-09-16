# Neuropoly — working notes

LightGBM pipeline predicting P(win) for Polymarket trades. CPU-only.

## Read this first

`docs/universe-and-grouping.md` carries the whole line of work: which markets
are trainable and why, what the diagnostics found, and — the part that matters
most — that **execution, not the model, decides profit**. It is written to be
picked up cold.

## Never run a heavy job unguarded

This box is an **LXC guest with no cgroup memory cap and no swap**, so a process
that overcommits takes the *host* down, not just itself. That has happened:
`run_pipeline.py` pushed MemAvailable from 15.4 to 4.4 GiB in six seconds and
the machine rebooted.

```bash
./runguard.sh --max-rss 10 --min-avail 14 -- .venv/bin/python -u <script> ...
```

Two independent limits, because they fail differently: `--max-rss` catches this
job's own growth, `--min-avail` catches the case where this job is innocent and
something else is eating memory. Poll every 2 s — at 5 s the crash above would
have been missed. It does **not** pass stdin, so pass a script path, not a
heredoc.

Stop daemons by **noted PID**, never `pkill -f` — a pattern that matches your
own command line kills the guard and leaves the compute running unguarded.

## Where the data comes from

`/root/poly_data` — a fork of `warproxxx/poly_data` **v2** on this same box.
**It collects; this repo joins, filters and trains.** It runs with
`PIPELINE_STAGES=collect`, so `processed/trades.csv` is never produced.

```
markets.csv  ──build_registry.py──▶  data/market_registry.parquet
orderFilled.csv ──build_trades.py─▶  data/trades.parquet      (universe only)
                 census_markets.py ▶  D0: what is trainable
                measure_spread.py  ▶  live order books
```

`convert_to_parquet.py` targets poly_data **v1** and is superseded. Do not
repair it — it fails silently, writing an empty `trades.parquet`.

## Environment

```bash
.venv/bin/python -m pytest tests/ -q      # 56 tests, all offline
```

`uv` lives at `/root/.local/bin/uv`. LightGBM needs `libgomp1` (apt) or every
training run dies with `libgomp.so.1: cannot open shared object file`.

## Invariants that break quietly

These have all bitten. None of them raise.

1. **One row group = one market** (`pipeline/rowgroups.py`). Every lag, rolling
   window and label is `.over("market_id")`; the longest window is 48 buckets,
   so a market shorter than that yields no complete feature row.
2. **`market_id` must stay unique.** poly_data's `markets.csv` can list a market
   twice, and joining against a duplicated prior registry multiplies rows.
   `build_registry.py` asserts uniqueness before writing.
3. **`market_id` must stay stable** across filter changes and export growth, or
   every Parquet built earlier points at the wrong markets. Ids are assigned to
   *all* markets and reused from the prior registry.
4. **`segments.py` is precision-tuned**, which is right for *selecting* a
   segment and wrong for *excluding* one. `pipeline/universe.py` widens the
   rules; if you touch them, measure **recall**.
5. **`taker_direction`, not `maker_direction`**, maps to `direction`. They are
   exact inverses; the wrong one flips `is_buy` for the whole dataset.
6. **Market identity is deliberately excluded from features**
   (`features.py:403`; `volume` was dropped for the same reason).
7. **Batch by rows, not by markets.** Markets differ in size by more than 10×
   (median 3,939 rows, max 47,754), so a market-count batch is not a memory
   bound — that is what made `add_labels_streaming` peak at 10.3 GiB.

## State

- Branch `claude/market-universe-filter`, pushed.
- **D0, D1, D2 done.** Per-market, per-family and per-segment models are all
  ruled out — see the summary table at the top of the docs.
- `model_d2_8h.txt` — global model, 8 h horizon, test AUC 0.6008.
- **Running now:** `collect_trades.py` + `paper_trades.py` under `runguard.sh`,
  measuring live maker fill rates on 42 markets. Needs ~3 days before most
  markets are scorable.

```bash
.venv/bin/python paper_trades.py --report --paper-db paper_trades.db
```

- **Next, ranked:** train on magnitude instead of direction; position sizing;
  then the fill measurement's verdict. `docs/universe-and-grouping.md` §6 has
  the full list, including what is already exhausted.

## Keep the documentation current — and ask before changing it

**When a finding is verified or a workflow changes, propose a documentation
update in the same breath, and get agreement before writing it.** Do not update
silently, and do not leave it for later.

*Verified* means measured and reproducible: a number that came out of a run, a
behaviour confirmed against the real data, a procedure that was actually
executed. Not a hunch, not a plausible explanation, not something that "should"
be true. Unverified things belong in the open-questions section, marked as such.

Where each kind of thing goes:

| finding | goes to |
|---|---|
| a measurement, a verdict, a closed question | `docs/universe-and-grouping.md` |
| an invariant that breaks quietly, a command, current state | `CLAUDE.md` |
| a changed data path or tool a user would run | `README.md` |
| something a future session must not re-derive | the memory files |

Two habits that follow from how this went wrong before:

* **Edit in place, do not append.** `docs/universe-and-grouping.md` once grew to
  two "D1 result" sections and two "D2 result" sections, one of each pair
  contradicting the other, plus a "remaining stages" list describing work that
  was already finished. A document that disagrees with itself is worse than one
  that is merely out of date.
* **Re-check every claim against the repo before writing it.** Test counts,
  file paths, line references and running processes all drift. Verify, then
  write.

When a result overturns something already written — including a recommendation
made earlier in the same session — say so explicitly and record the reason. The
reversal is usually the most useful part.

## Conventions

- Tests use **real slugs** from the export, never invented ones — a made-up
  slug proves nothing about a regex written against production naming.
- Every test file runs standalone (`python tests/test_x.py`) and under pytest.
- Docstrings explain *why*, with the measured number that motivated the rule.
