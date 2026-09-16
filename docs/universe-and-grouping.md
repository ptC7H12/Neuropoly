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

## D0 result (2026-09-16) — and what it settles

`census_markets.py`, run on 33,042,239 events across 155,081 markets that
actually traded.

**Trainable: 8,767 markets — 5.7 % of those that traded, but 88.6 % of USD
volume.** The shallow markets are numerous and economically irrelevant. Volume
is the headline number precisely because market counts mislead here.

Verdict against the stopping rule: 88.6 % >> 20 %, so the feature architecture
is *not* the constraint. Proceed.

Measuring `active` and `span` separately was what made this honest. 70,177
markets (45.3 %) have a calendar span of >= 54 buckets, but only 8,767 have 54
buckets that really traded — an 8x overstatement if only span were counted.
Median active buckets per market is **3**.

Upper bound on feature rows: **2,182,160**.

### This kills the per-family idea

Family structure *within the trainable subset* is nothing like the universe:

| Family size | Families | Markets | Share | Volume |
|---|---|---|---|---|
| >= 1000 | **0** | 0 | 0.0 % | — |
| 100-999 | 1 | 111 | 1.3 % | 0.00B |
| 10-99 | 83 | 1,445 | 16.5 % | 1.04B |
| 2-9 | 1,023 | 3,202 | 36.5 % | 1.14B |
| singleton | 4,009 | 4,009 | 45.7 % | 0.92B |

The 60 families with >= 1000 members counted earlier were the candle-like
shallow markets — exactly the ones that are not trainable. The largest
trainable family is `will-donald-trump-publicly-insult-someone-on`, with 111
markets; 45.7 % are one-offs.

**So there is no family with enough trainable markets to carry its own model**
with a walk-forward split that has a non-empty val and test. D4 (models per
family) is off the table, and `family` as a categorical feature over 5,116
levels — half of them singletons — would be an overfitting surface, not a
signal.

What remains as a grouping axis is `segment`, and in this universe that is two
values: `other` (150,552 markets, 2.44B, 83.8 % of its volume trainable) and
`politics` (4,529 markets, 1.08B, **99.4 %**). Volume is concentrated in
geopolitical event markets — Iran/US, Russia/Ukraine, Strait of Hormuz — which
are 100 % trainable and worth hundreds of millions each.

D1 should therefore compare `other` against `politics`, and consider coarse
volume/depth strata, not families.

## D1 result (2026-09-16) — and what it settles

`sweep_horizon.py` on `data/trades_trainable.parquet` (8,767 markets), six
horizons, `--by-segment`. Cost model at the default `spread_abs=0.01`.

| Horizon | Labeled | MedianRet | MedianCost | Ret>Cost | MaxROI* |
|---|---|---|---|---|---|
| 30 min | 890,386 | 0.329 % | 8.82 % | **17.38 %** | +37.90 % |
| 1 h | 851,684 | 0.484 % | 8.67 % | 21.56 % | +43.53 % |
| 2 h | 803,323 | 0.689 % | 8.36 % | 26.41 % | +45.39 % |
| 4 h | 740,424 | 0.924 % | 7.62 % | 30.89 % | +48.40 % |
| 8 h | 677,677 | 1.149 % | 6.75 % | 36.50 % | +58.13 % |
| **1 d** | 602,561 | 1.818 % | 5.16 % | **47.93 %** | +65.23 % |

### The finding that is not about grouping at all

`Ret>Cost` rises monotonically with the horizon, from 17.4 % to 47.9 %, and
`MedianCost` falls from 8.82 % to 5.16 %. **`LabelConfig.forward_window_buckets`
defaults to 6 — 30 minutes — which is the worst horizon tested.** Moving to a
day nearly triples the share of buckets whose move pays for the round trip.

Longer horizons also select for the deeper markets, which is why cost falls:
a market that survives 288 buckets is a liquid one.

### By segment

| | politics 30 min | other 30 min | politics 1 d | other 1 d |
|---|---|---|---|---|
| Labeled | 233,667 | 656,719 | 182,344 | 420,217 |
| MedianRet | 0.947 % | 0.592 % | 5.000 % | 2.992 % |
| MedianCost | 8.41 % | 9.06 % | 5.64 % | 5.97 % |
| Ret>Cost | 12.27 % | 19.19 % | **48.88 %** | **47.52 %** |

`politics` carries a lower median cost at every horizon, as expected — its
volume sits in a few deep geopolitical markets. Its median return is also
~60 % higher throughout.

**At the horizon worth trading, the two segments are indistinguishable**:
`Ret>Cost` 48.88 % against 47.52 % (2.8 % relative), `MedianCost` 5.64 %
against 5.97 %. They diverge only at 30 minutes — a horizon the table above
rules out anyway.

Note the apparent paradox at 30 min: politics has both a *higher* median
return and a *lower* median cost, yet a *lower* `Ret>Cost`. Medians are taken
over different buckets, so the two do not compose; politics must carry a more
skewed return distribution, with more buckets sitting below their own
individually higher cost. Worth understanding before leaning on the segment
split for anything.

### Verdict: the grouping question is closed

Per the stopping rule, `Ret>Cost` and `MedianCost` do **not** differ materially
between segments at the viable horizon. Combined with D0 — which showed no
family has enough trainable markets to carry a model — the ladder collapses:

* **L3 per market** — dead (D0: median 3 active buckets).
* **L2 per family** — dead (D0: largest trainable family has 111 markets,
  45.7 % are singletons).
* **L1 per segment** — not warranted (D1: the segments converge at 1 d).
* **L0 one global model** — what remains.

D2 and D4 are therefore moot. **D3** — `segment` as a categorical feature in
one model — stays worth the ~15 lines, since it costs no data fragmentation
and lets LightGBM use the split if it helps at all.

The open work is no longer *which grouping* but **the label horizon**, which
D1 turned up on the way past.

## D1 result (2026-09-16) — the horizon matters more than the grouping

`sweep_horizon.py` over the 8,767 trainable markets, six horizons, by segment.
Costs at the default `spread_abs = 0.01`.

| Horizon | Labeled | MedianRet | MedianCost | Ret>Cost | MaxROI* |
|---|---|---|---|---|---|
| 30 min | 890,386 | 0.329 % | 8.82 % | 17.38 % | +37.90 % |
| 1 h | 851,684 | 0.484 % | 8.67 % | 21.56 % | +43.53 % |
| 2 h | 803,323 | 0.689 % | 8.36 % | 26.41 % | +45.39 % |
| 4 h | 740,424 | 0.924 % | 7.62 % | 30.89 % | +48.40 % |
| 8 h | 677,677 | 1.149 % | 6.75 % | 36.50 % | +58.13 % |
| **1 d** | 602,561 | 1.818 % | 5.16 % | **47.93 %** | +65.23 % |

**The headline is not the segment split, it is the holding period.** `Ret>Cost`
nearly triples from 30 minutes to a day, while the config default
(`forward_window_buckets = 6`) sits on the weakest row in the table. Returns
grow with the horizon and costs shrink, because a longer hold reaches deeper,
better-priced buckets.

For scale: the example table in `README.md` shows 0.36 % at 30 minutes. This
universe gives 17.38 % at the same horizon — the filtering work is what bought
that, and it is the clearest validation of it so far.

### Segments differ at short horizons and converge at long ones

| Horizon | politics Ret>Cost | other Ret>Cost | politics MedianCost | other MedianCost |
|---|---|---|---|---|
| 30 min | 12.27 % | 19.19 % | 8.41 % | 9.06 % |
| 4 h | 27.47 % | 32.23 % | 7.25 % | 7.90 % |
| 1 d | 48.88 % | 47.52 % | 5.64 % | 5.97 % |

A paradox worth understanding before acting on it: `politics` has a **higher**
median return *and* **lower** median cost at every horizon, yet a **worse**
`Ret>Cost` everywhere except one day. `Ret>Cost` pairs each bucket's return
against that same bucket's cost, so distribution shape decides it, not the
medians.

Measured explanation: **45.0 %** of politics trades happen at extreme prices
(below 0.10 or above 0.90) against **36.0 %** for `other`. Cost scales with
`1/price`, so on the cheap side the politics p90 round-trip is **119.1 %** of
the position value — the trade costs more than it can pay — against 84.9 % for
`other`.

### Verdict

`MedianCost` differs by at most 0.7 pp between segments, and `Ret>Cost`
converges by one day. Against the stopping rule that is not enough to justify
separate models, and `politics` carries only ~25 % of labels, so splitting
would starve it. **Go to D3: `segment` as a categorical feature in one model.**

But do the horizon first. Moving `forward_window_buckets` from 6 towards 96-288
is a larger lever than any grouping decision in this document, and it is a
one-line config change.

**Caveat that governs all of the above:** `spread_abs = 0.01` is the single
empirical input and the quartiles across 120 live order books are
0.001 / 0.010 / 0.039. Re-run with the value actually seen in the markets to be
traded before treating any of these numbers as final.

## D2 result (2026-09-16) — the grouping question is closed

One model, one split, 8 h horizon, scored separately on each segment's test rows.

| segment | rows | AUC | Brier | WinRate | Profit% | ROI | Sharpe | trades |
|---|---|---|---|---|---|---|---|---|
| other | 66,462 | 0.6027 | 0.241 | 66.3 % | 27.9 % | −1.4 % | 0.08 | 11,855 |
| politics | 16,870 | 0.5933 | 0.244 | 63.1 % | 25.8 % | −2.0 % | 0.06 | 2,897 |

**The AUC gap is 0.0094.** Against the stopping rule that is "streut kaum" —
one model already covers both segments equally well. Separate models are not
justified, and D3 has little left to win either: a categorical feature can only
help where the groups actually behave differently, and here they do not.

### The original question, answered at three levels

| Level | Verdict | Where measured |
|---|---|---|
| per market | impossible — median market has 3 active buckets | D0 |
| per family | impossible — no trainable family reaches 1000 markets, largest is 111, 45.7 % are singletons | D0 |
| per segment | not worth it — costs converge by 1 d, AUC differs by 0.009 | D1, D2 |

## The real obstacle is cost, not the model

The global backtest: **ROI −1.55 %** on 14,752 trades. Win rate 65.7 %, but
only 27.5 % profitable after fees. Mean return per trade **+3.82 %** against
mean cost **5.37 %** — the 1.55 pp shortfall *is* the ROI.

So the model discriminates (AUC 0.60, two thirds of trades move the right way)
and still loses money, exactly the failure mode `README.md` warns about: it
gets the direction right, and the moves do not carry the costs.

Two consequences worth acting on before more modelling:

1. **`spread_abs` is an assumption, not a measurement.** Everything above uses
   0.01, the median of 120 live order books whose quartiles are
   0.001 / 0.010 / 0.039. At the lower quartile the cost side of that
   comparison collapses and the sign of the ROI flips. Measuring the real
   spread in the markets actually traded is now the single highest-value work
   left — larger than any modelling change.

2. **The training target does not match the economic objective.** `win` is
   binary: did the price move at least `min_price_move` (0.001) the right way.
   A move of 0.001 and a move of 0.05 are the same label. The model therefore
   optimises *direction*, while profitability needs *magnitude*. Raising the
   entry threshold selects for confident direction, not for large moves, so it
   cannot fix this on its own. A magnitude-aware label — or a regression on
   `trade_return` — addresses it directly.

## D2 result (2026-09-16) — the grouping question is answered

One model, one split, one set of cut times, scored separately per segment.
8-hour horizon (`forward_window_buckets = 96`), 555,526 labeled rows, split
387,380 train / 82,461 val / 83,332 test. Global test ROC AUC 0.6008.

| segment | rows | AUC | Brier | WinRate | Profit% | ROI | Sharpe | trades |
|---|---|---|---|---|---|---|---|---|
| other | 66,462 | 0.6027 | 0.241 | 66.3 % | 27.9 % | −1.4 % | 0.08 | 11,855 |
| politics | 16,870 | 0.5933 | 0.244 | 63.1 % | 25.8 % | −2.0 % | 0.06 | 2,897 |

**The AUC gap is 0.0094.** Against the stopping rule that is as flat as it
gets: the two segments are equally predictable, so separate models are not
justified — and D3 has little to gain either, since there is no group-specific
structure for a categorical feature to expose.

### The original question, closed

| Granularity | Verdict | Why |
|---|---|---|
| per market | impossible | D0: median 3 active buckets; 8,767 of 155,081 trainable |
| per family | impossible | D0: no trainable family reaches 1000 markets, largest is 111, 45.7 % singletons |
| per segment | not justified | D2: AUC 0.6027 vs 0.5933 |
| **global** | **use this** | — |

### The finding that matters more

Both segments show **negative ROI** while the model plainly has signal: AUC
0.60 out of sample with time-purged splits, and a win rate of 66.3 %.

Win rate 66.3 % against **27.9 % profitable** is exactly the gap `README.md`
warns about. The model gets the direction right two times in three, and the
moves are still too small to clear a ~7 % round trip. Direction is not the
problem; magnitude is.

Levers, in measured order of size:

1. **Horizon.** D1: `Ret>Cost` is 36.5 % at 8 h and 47.9 % at one day. This
   run sits at 8 h. One config line.
2. **Entry threshold.** 0.6 today. Higher means fewer, higher-conviction
   buckets — which is the right direction when the binding constraint is that
   the average qualifying move is too small.
3. **`spread_abs`.** Set to 0.01, the median across 120 live order books, but
   the quartiles are 0.001 / 0.010 / 0.039. At the lower quartile the cost
   model changes shape entirely. This is an assumption, not a measurement of
   the markets actually traded, and it deserves to become one.

## Spread measured (2026-09-16) — the assumption held, and break-even is close

`measure_spread.py` pulled the live CLOB order book for every still-open
trainable market: **1,685 books in 12 seconds.** Of 8,767 trainable markets
1,880 are still open and all of them have a book.

| | n | median | p25 | p75 | p90 |
|---|---|---|---|---|---|
| all | 1,685 | **0.0100** | 0.0100 | 0.0300 | 0.0600 |
| other | 1,264 | 0.0110 | 0.0100 | 0.0300 | 0.0770 |
| politics | 421 | 0.0100 | 0.0090 | 0.0270 | 0.0470 |

Volume-weighted: **0.0083**. The 0.01 assumption every earlier number used was
right, so the hoped-for escape — "maybe our markets are far tighter" — is
closed. Note also that 1,192 of these markets have a 0.01 tick size, so for
them 0.01 is the *floor*, not an average.

### What that does to the ROI

The saved model re-scored at different spreads and entry thresholds — no
retraining, same split, same trees:

| spread \ threshold | 0.60 | 0.63 | **0.65** | 0.68 | 0.70 |
|---|---|---|---|---|---|
| 0.0100 (assumed) | −1.55 % | −0.99 % | −0.64 % | −1.31 % | −1.70 % |
| **0.0083 (measured)** | −1.11 % | −0.57 % | **−0.23 %** | −0.88 % | −1.42 % |
| 0.0060 | −0.60 % | −0.12 % | **+0.17 %** | −0.53 % | −1.13 % |
| 0.0040 | −0.17 % | +0.29 % | **+0.50 %** | −0.28 % | −0.91 % |
| 0.0030 | +0.09 % | +0.50 % | **+0.69 %** | −0.11 % | −0.76 % |

Two independent levers, and both matter:

* **Entry threshold 0.65 is the optimum at every spread.** Below it, too many
  marginal trades; above it, the realised return collapses (1.44 % at 0.70,
  negative at 0.75) faster than the cost falls. At 0.65 the model trades 7,585
  times instead of 14,759 and the mean cost drops from 5.00 % to 3.84 %,
  because its confident picks sit in higher-priced tokens that are cheaper to
  trade.
* **Break-even in spread sits near 0.007.** Measured median is 0.0100 and
  volume-weighted 0.0083, so the gap is roughly a factor of 1.2-1.4 — not the
  factor of 10 that would need a different market.

**And the tight markets exist.** 262 of the 1,685 books (15.5 %) quote a spread
of 0.005 or less, and they carry 165.4 M USD of the volume. Restricting
execution to those puts the backtest at roughly +0.2 % to +0.5 %.

### Caveats that matter more than the numbers

* **The threshold was read off the test split.** Picking 0.65 by looking at
  test ROI is selection on the evaluation set. It has to be chosen on the
  validation split before any of these figures can be claimed out-of-sample.
* **+0.5 % ROI is thin.** It is not zero and it is not a business either; it
  leaves no room for slippage beyond the top of book, and depth was recorded
  but not yet used.
* **Fees alone cost about 3 %** of the position at threshold 0.65. That is the
  floor no execution improvement can go below.

The next move is therefore an *execution* change, not a modelling one: make the
live spread a trading precondition, and re-pick the threshold on validation.

## The diagnostic (remaining stages)

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

### Trades: read the raw events, not poly_data's stage 3

**This reverses the earlier recommendation in this document, on measurement.**
The split is now: poly_data collects, Neuropoly joins, filters and trains.
`run_pipeline.sh` defaults to `PIPELINE_STAGES=collect` (markets + chain only)
and `build_trades.py` here does the rest.

Two independent reasons, both measured:

* **Size.** orderFilled.csv is ~163 GB (~640 M rows). `processed/trades.csv`
  adds a 66-char hex `market_id` and an ISO timestamp per row, landing near
  ~195 GB against 270 GB free — and ~73 % of it would be discarded here.
* **Memory.** `process_live._discover_missing_tokens()` runs *before* the
  chunked loop and ignores `PROCESS_CHUNK_SIZE`: two `.unique().collect()`
  passes over the whole file, then Python sets of 77-char strings. It tripped
  the RAM watchdog on 2026-09-16 and stopped collection for ten hours.

`build_trades.py` joins against the registry with `how="inner"`, so the
universe filter applies *during* the scan. Measured on the first 2 M events:
**122,309 rows survive — 6.1 %, a 16x reduction**, far better than the 27 %
market-count share suggests, because the excluded candle instruments are
hyperactive.

Two open questions from earlier are now closed:

* **Gamma backfill loss: 0 %.** Not one event in 2 M referenced a token absent
  from the whole registry. `build_trades.py --count-unmatched` re-measures it.
* **No dedup needed.** `convert_to_parquet.py` assumes "each trade appears
  TWICE"; that is false for v2. Every event carries exactly one USDC leg
  (1,736,203 maker-side + 263,797 taker-side = 2,000,000), and only 1.2 % of
  output rows repeat exactly — genuine partial fills at one price in one
  transaction, not duplicated legs.

The column mapping below still describes `processed/trades.csv`, kept for the
case where `PIPELINE_STAGES=full` is used. `build_trades.py` emits the stock
`config.py` column names directly, so no mapping is needed on that path.

### Column mapping for processed/trades.csv (legacy path)

Configurable in `config.py` except `market_id`:

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
