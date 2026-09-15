# Trainable universe and per-group training

Status: 2026-09-15. Written so this work can be picked up cold.

## The question

Does it pay to train **separate models per market**, because bidding behaviour
differs from market to market?

It is meant to be *measured*, not assumed. This repo already has a precedent
for that kind of decision: `sweep_horizon.py` answers "does the holding period
even pay for its costs?" **before** a model is built. This document extends the
same discipline one level up.

## Why the question had to be reframed first

Every lag and rolling feature is computed `.over("market_id")` — see the
invariant in `pipeline/rowgroups.py` — with windows up to
`max(rolling_windows) = 48` buckets, plus `forward_window_buckets = 6` for the
label. At `bucket_minutes = 5` a market must trade for well over four hours
before it yields one complete feature row.

Polymarket is dominated by scheduled short-runners. A `btc-updown-5m` market
lives 300 seconds — **exactly one bucket**. Those markets do not produce
*little* data, they produce **none**.

And pooling them by family does not help: the window functions restart per
`market_id` regardless of how markets are grouped. Making them trainable would
mean treating a recurring template as one continuous series, which is a
redesign of the `.over("market_id")` invariant, not a config change.

So the first question is not "which grouping" but **"which markets are
trainable at all"**.

## The universe (decided 2026-09-09)

Excluded, in this order — the order is part of the definition, because the
first rule that fires names the reason:

| Reason | Kind | Why |
|---|---|---|
| `periodic` | structural | scheduled candles; yield zero complete feature rows |
| `sports` | thematic | results repricing, not belief revision |
| `crypto` | thematic | mechanical price bets on an external feed |
| `asset_price` | thematic | equities, indices, CS:GO skins — same template as crypto |
| `short_lived` | structural | under 12 h (3x the longest feature window) |

Roughly **27 % of ~3.4 M markets survive** (~910 k), and they are the
long-lived ones: non-templated markets have a median life of ~6 days against
~5 minutes for the candles.

Implementation: `pipeline/universe.py` (rules), `pipeline/families.py`
(slug → family), `build_registry.py` (CLI + report).

### The recall inversion — the subtlest part

`pipeline/segments.py` is tuned for **precision**: its docstring says "anything
unmatched becomes `other`, never a subject segment". Measured against Gamma
tags: precision 100 %, recall 97.4 % (sports), 72.5 % (crypto), 84.0 %
(politics).

Selecting a segment cares about precision. **Excluding** one cares about
recall — every miss stays in the training set. So `universe.py` deliberately
widens both rules and reports the delta under `--validate`.

This is not theoretical. The sports fixture pattern in `segments.py` spells
team codes `[a-z]{2,4}`, so it misses every roster with a digit. That left
**86,014 esports and football fixtures — 9.1 % of the surviving universe** — in
the training set: efa 25,663 · lol 17,715 · itf 12,345 · val 8,133, plus
crint, cwbb, bun, dfb, ahl, codmw. A 25-slug random sample was sports without
exception.

**If you retune the filter, re-measure recall, not precision.**

## Granularity ladder

| Level | Unit | Notes |
|---|---|---|
| L0 | global | one model, status quo |
| L1 | segment | `sports`/`crypto`/`politics`/`other` — machinery exists, but **93 % of the surviving universe is `other`**, so segment is nearly useless as a grouping key here |
| L2 | **family** | slug with digits/months masked — the meaningful unit |
| L3 | single market | only for the long-lived, high-volume head |

Two family keys are produced, deliberately: `family` (masked slug, keeps
window length as identity) and `family_root` (trailing slots stripped, merges
date variants). Which granularity supports separate models is for the
diagnostic to answer.

### Measured family distribution in the universe

| Family size | Families | Markets | Share |
|---|---|---|---|
| >= 1000 | 60 | 116,433 | 12.4 % |
| 100-999 | 152 | 38,183 | 4.1 % |
| 10-99 | 7,397 | 128,371 | 13.6 % |
| 2-9 | 125,384 | 391,845 | 41.6 % |
| singleton | 267,285 | 267,285 | 28.4 % |

**Only 60 families have >= 1000 members.** 70 % of markets sit in families of
<= 9, where a per-family model is impossible from the start. So "models per
group" reduces to those ~60 families plus one pooled remainder.

Largest surviving families: `elon-musk-of-tweets` (4,766), then weather —
`highest-temperature-in-{seattle,london,dallas,atlanta,nyc,...}-on` (~2,400-2,650
each). Weather is ~14 % of the universe.

## The diagnostic (not yet built)

Five stages, each may stop the next. **Blocked on `processed/trades.csv`**,
which poly_data's stage 3 has not yet produced.

- **D0 — trainability census** (`census_markets.py`, to write). Bucket counts
  per market; how many reach >= 54 buckets (48 + 6); share of *volume* in
  non-trainable markets; how many trade rows the universe filter saves. Reuse
  `rowgroups.iter_market_row_groups()` and `aggregation.aggregate_trades()`.
- **D1 — cost floor per group.** Extend `sweep_horizon.py` with `--group-col`.
  ~5 lines: `seg_accs` is already a dict over arbitrary keys; make it a
  `defaultdict(Accumulator)` and generalise `segment_of()`.
- **D2 — one global model, error decomposed by group.** Not possible today:
  `SplitResult` (`pipeline/splitter.py`) carries no `market_id`. Add
  `train_mid/val_mid/test_mid` (~8 lines), then report AUC/Brier/ROI per group
  on the test split. **This answers the user's question most directly.**
- **D3 — group as a categorical feature.** There is not one
  `categorical_feature` call in the repo; the path `README.md` itself
  recommends was never built. Join the group in `_add_market_features`, drop it
  from the `exclude` set in `get_feature_columns`, cast to `pl.Categorical`,
  pass `categorical_feature=[...]` in `model.py`. ~15 lines, one model, no data
  fragmentation.
- **D4 — models per group.** Only if D3 is not enough, and only for
  data-rich groups.

### Decision rule

```
D0: < 20 % of volume trainable?      -> STOP. Feature architecture is the problem.
D1: Ret>Cost / MedianCost flat?      -> STOP. One global model.
D2: test AUC flat across groups?     -> STOP. Homogeneous enough.
D3: beats the D2 baseline?           -> DONE. Categorical feature, one model.
else D4.
```

### The trap in D4

Split boundaries are **row-count percentiles** (`splitter.py:95-101`). Computed
per group they land on *different calendar dates*, so per-group results are
comparable neither to each other nor to the global model. Thread the globally
computed cut *times* through instead — add an optional `cut_times` parameter.
`benchmark_strategies.py:707-751` already does the right thing (split globally,
then filter test rows) and is the template.

Also: `walk_forward_split` raises on an empty split — catch and skip per group.
`TrainingMonitor` opens its log with `"w"` — give each group its own path. And
warm-start each group model from the global one (`train_chunked.py:299` shows
the pattern), which blunts the data fragmentation the README warns about.

## The data bridge: poly_data -> Neuropoly

poly_data v2 (`/root/poly_data`, fork of `warproxxx/poly_data`) produces
`data/markets.csv`, `data/orderFilled.csv` and `processed/trades.csv`.

Neuropoly's `convert_to_parquet.py` still expects poly_data **v1** and is
superseded by `build_registry.py` — it is deliberately *not* repaired.

### Trades mapping

Feed `processed/trades.csv`, not the raw files. Configurable in
`config.py` except `market_id`:

| poly_data | Neuropoly | How |
|---|---|---|
| `timestamp` | `timestamp` | default |
| `market_id` (hex) | `market_id` (Int32) | **join via the registry** |
| `nonusdc_side` | `side` | `trades_side_col="nonusdc_side"`; values `token1`/`token2` already match `side_yes`/`side_no` |
| `taker_direction` | `direction` | `trades_direction_col="taker_direction"` |
| `price` | `price` | same convention; `data_loader.py` flips NO -> P(YES) |
| `usd_amount`, `token_amount` | same | already 1e6-scaled |

> **Trap:** `maker_direction` is the exact inverse of `taker_direction`.
> Choosing it inverts `is_buy` for the entire dataset, silently.
> `convert_to_parquet.py:264` calls its column "from maker's view" but line 408
> computes the **taker's** view — the docstring is wrong, the code is right.

Still needed in code: drop the `market_id -> Int64` cast in
`data_loader.py:80,147`, pre-filter `market_id.is_not_null()` and
`price` in (0,1) (poly_data divides without a zero guard), and decide a
dedup policy for multi-leg fills (`transactionHash` is the key).

### Why a dense Int32 id

poly_data identifies markets by `condition_id` (66-char hex); this pipeline
casts `market_id` to Int64 in eleven places. Both failure modes are **silent**:
`convert_to_parquet._build_token_lookup` skips every market and writes an empty
`trades.parquet`; `segments.segment_of` catches the ValueError and labels all
3.4 M markets `other`.

The registry assigns a dense Int32 id and carries the hex alongside, so every
existing cast stays valid — and 4 bytes per row instead of 66 matters at 100 M+
trade rows on a 62 GB box.

**Ids are assigned to every market, not only kept ones, and a prior registry is
reloaded.** Both exist so `market_id` survives a retuned filter and a grown
export. The universe is a *view* (`keep` / `exclude_reason`), never a subset.

### markets.csv contains duplicates

poly_data's `update_markets` resume path dedups only against the last ~1500
rows (`_read_tail_ids`), so a market re-fetched after a restart is appended
again. Measured: 25,464 duplicates. poly_data itself reads around this
(`get_markets()` does `.unique(subset=["id"], keep="first")`) and the registry
does the same — on the input **and** on the prior registry it joins against,
because a duplicated join partner *multiplies* rows. `build_registry.py` now
asserts uniqueness before writing.

## Open questions

- Weather markets are ~14 % of the universe. Kept for now; nobody has decided
  whether they belong.
- `will-sp500-close-above-6000` (prose) is kept while `spx-above-6000-on-...`
  (templated) is excluded. The anchor is deliberate but the asymmetry is real.
- The `crypto` reason removes crypto-*themed* markets too (e.g. "will Coinbase
  be acquired"), not only price bets. Conservative on purpose.
