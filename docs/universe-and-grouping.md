# Trainable universe, grouping, and what actually blocks profit

Status: 2026-09-16. Written so this work can be picked up cold.

## The question, and the answer

**Does it pay to train separate models per market, because bidding behaviour
differs from market to market?**

**No — measured at three levels, each with a different reason:**

| level | verdict | measured in |
|---|---|---|
| per market | impossible — the median traded market has **3** active 5-minute buckets | D0 |
| per family | impossible — no trainable family reaches 1000 markets, the largest has 111, 45.7 % are one-offs | D0 |
| per segment | not worth it — costs converge by a one-day horizon, test AUC differs by **0.009** | D1, D2 |

What the work surfaced instead is more useful: **the model is not the problem.**
It gets direction right two thirds of the time. What decides profit is
execution — the spread, the fee legs, and whether a resting order fills.

## Why the question had to be reframed first

Every lag and rolling feature is computed `.over("market_id")` — see the
invariant in `pipeline/rowgroups.py` — over windows up to
`max(rolling_windows) = 48` buckets, plus `forward_window_buckets` for the
label. At 5-minute buckets a market must trade for hours before it yields one
complete feature row.

Polymarket is dominated by scheduled short-runners. A `btc-updown-5m` market
lives 300 seconds — **exactly one bucket**. Those markets produce not little
data but **none**, and pooling them by family does not help: the windows
restart per `market_id` regardless of grouping.

So the first question was never "which grouping" but **"which markets are
trainable at all"**.

---

# 1. The trainable universe

Excluded in this order — the order is part of the definition, because the first
rule that fires names the reason:

| reason | kind | markets | share |
|---|---|---|---|
| `periodic` | structural — scheduled candles yield no complete rows | 696,070 | 20.6 % |
| `sports` | thematic — results repricing, not belief revision | 1,403,453 | 41.4 % |
| `crypto` | thematic — mechanical bets on an external feed | 191,254 | 5.7 % |
| `asset_price` | thematic — equities, indices, CS:GO skins, same template | 7,502 | 0.2 % |
| `short_lived` | structural — under 12 h, 3× the longest feature window | 180,458 | 5.3 % |
| **kept** | | **907,991** | **26.8 %** |

Rules in `pipeline/universe.py`, families in `pipeline/families.py`, built by
`build_registry.py`.

## The recall inversion — the subtlest part of this

`pipeline/segments.py` is tuned for **precision**: "anything unmatched becomes
`other`, never a subject segment". Measured against Gamma tags: precision
100 %, but recall **97.4 %** (sports), **72.5 %** (crypto), **84.0 %**
(politics).

Selecting a segment cares about precision. **Excluding** one cares about
recall — every miss stays in the training set. So `universe.py` deliberately
widens both rules and reports the delta under `--validate`.

This was not theoretical. The fixture pattern in `segments.py` spells team
codes `[a-z]{2,4}` and so misses every roster containing a digit, which left
**86,014 esports and football fixtures — 9.1 % of the surviving universe** — in
the training set: efa 25,663 · lol 17,715 · itf 12,345 · val 8,133, plus crint,
cwbb, bun, dfb, ahl, codmw. A 25-slug random sample was sports without
exception.

**If you retune the filter, measure recall, not precision.**

---

# 2. What the diagnostics found

## D0 — trainability census (`census_markets.py`)

Run on 33,042,239 events across 155,081 markets that actually traded.

**Trainable: 8,767 markets — 5.7 % by count, but 88.6 % of USD volume.** The
shallow markets are numerous and economically irrelevant, which is why volume
is the headline number and market counts mislead.

Counting `active` and `span` separately is what makes this honest: 70,177
markets (45.3 %) span ≥ 54 buckets on the calendar, but only 8,767 have 54
buckets that *really traded* — an eightfold overstatement if only span were
counted. Median active buckets per market: **3**. Upper bound on feature rows:
**2,182,160**.

### This is what killed the per-family idea

Family structure inside the trainable subset is nothing like the universe:

| family size | families | markets | share |
|---|---|---|---|
| ≥ 1000 | **0** | 0 | 0.0 % |
| 100–999 | 1 | 111 | 1.3 % |
| 10–99 | 83 | 1,445 | 16.5 % |
| 2–9 | 1,023 | 3,202 | 36.5 % |
| singleton | 4,009 | 4,009 | 45.7 % |

The 60 families with ≥ 1000 members counted over the whole universe were the
shallow candle-like markets — exactly the ones that cannot be trained on. No
family carries enough trainable markets for its own walk-forward split, and
`family` as a categorical over 5,116 mostly-singleton levels would be an
overfitting surface, not a signal.

Segments in the trainable set: `other` 150,552 markets / 2.44 B volume /
83.8 % of it trainable; `politics` 4,529 / 1.08 B / **99.4 %**.

## D1 — cost floor (`sweep_horizon.py --by-segment`)

8,767 trainable markets, six horizons, `spread_abs = 0.01`.

| horizon | labeled | MedianRet | MedianCost | Ret>Cost | MaxROI* |
|---|---|---|---|---|---|
| 30 min | 890,386 | 0.329 % | 8.82 % | 17.38 % | +37.90 % |
| 1 h | 851,684 | 0.484 % | 8.67 % | 21.56 % | +43.53 % |
| 2 h | 803,323 | 0.689 % | 8.36 % | 26.41 % | +45.39 % |
| 4 h | 740,424 | 0.924 % | 7.62 % | 30.89 % | +48.40 % |
| 8 h | 677,677 | 1.149 % | 6.75 % | 36.50 % | +58.13 % |
| **1 d** | 602,561 | 1.818 % | 5.16 % | **47.93 %** | +65.23 % |

**The holding period is a bigger lever than any grouping.** `Ret>Cost` nearly
triples from 30 minutes to a day while `forward_window_buckets = 6` — the
config default — sits on the weakest row. For scale, `README.md`'s example
table shows 0.36 % at 30 minutes; this universe gives 17.38 %. That difference
is what the filtering bought.

Longer horizons cost trainable data, but far less than the market count
suggests:

| horizon | buckets needed | markets | feature rows | volume | Ret>Cost |
|---|---|---|---|---|---|
| 30 min | 54 | 8,767 | 2,182,160 | 88.6 % | 17.4 % |
| 4 h | 96 | 4,827 | 1,915,512 | 81.3 % | 30.9 % |
| **8 h** | 144 | 3,311 | 1,726,517 | 77.9 % | 36.5 % |
| 1 d | 336 | 1,446 | 1,315,501 | 66.6 % | 47.9 % |

8 h was chosen for D2: double the signal of the default while keeping 3,311
markets for generalisation.

### Segments differ at short horizons and converge at long ones

| horizon | politics Ret>Cost | other Ret>Cost | politics cost | other cost |
|---|---|---|---|---|
| 30 min | 12.27 % | 19.19 % | 8.41 % | 9.06 % |
| 4 h | 27.47 % | 32.23 % | 7.25 % | 7.90 % |
| 1 d | 48.88 % | 47.52 % | 5.64 % | 5.97 % |

An apparent contradiction worth understanding: `politics` has a **higher**
median return *and* **lower** median cost at every horizon, yet a **worse**
`Ret>Cost` everywhere except one day. `Ret>Cost` pairs each bucket's return
against that same bucket's cost, so distribution shape decides it, not medians.

Measured cause: **45.0 %** of politics trades happen at extreme prices (below
0.10 or above 0.90) against **36.0 %** for `other`. Cost scales with `1/price`,
so on the cheap side the politics p90 round-trip is **119.1 %** of the position
value — more than the trade can pay — against 84.9 % for `other`.

## D2 — one model, scored per segment (`run_pipeline.py --by-segment`)

One model, one split, one set of cut times, 8 h horizon. Any spread in these
numbers is a property of the markets, not of how they were fitted.

| segment | rows | AUC | Brier | WinRate | Profit% | ROI | Sharpe | trades |
|---|---|---|---|---|---|---|---|---|
| other | 66,462 | 0.6027 | 0.241 | 66.3 % | 27.9 % | −1.4 % | 0.08 | 11,855 |
| politics | 16,870 | 0.5933 | 0.244 | 63.1 % | 25.8 % | −2.0 % | 0.06 | 2,897 |

**AUC gap: 0.0094.** One model already covers both. Separate models are
unjustified, and D3 (segment as a categorical feature) has little left to win —
a categorical helps only where groups behave differently.

Scoring by group at all required a change: `SplitResult` carried no
`market_id`, so a global model's error could not be decomposed after the fact.
`walk_forward_split` now fills `train_mid` / `val_mid` / `test_mid`.

---

# 3. Why it loses money, and what moves it

Global backtest at the defaults: **ROI −1.55 %** on 14,752 trades, win rate
65.7 %, only 27.5 % profitable. Mean return **+3.82 %** against mean cost
**5.37 %** — that 1.55 pp shortfall *is* the ROI. The model gets the direction
right and the moves do not carry the costs.

## The spread was measured, and the assumption held

`measure_spread.py` pulled the live CLOB book for every still-open trainable
market: **1,685 books in 12 seconds**.

| | n | median | p25 | p75 | p90 |
|---|---|---|---|---|---|
| all | 1,685 | **0.0100** | 0.0100 | 0.0300 | 0.0600 |
| other | 1,264 | 0.0110 | 0.0100 | 0.0300 | 0.0770 |
| politics | 421 | 0.0100 | 0.0090 | 0.0270 | 0.0470 |

Volume-weighted **0.0083**. The 0.01 assumption was right, so the hoped-for
escape — "maybe our markets are far tighter" — is closed. 1,192 of these have a
0.01 tick size, where 0.01 is the *floor*, not an average.

Re-scoring the saved model across spread and entry threshold (no retraining):

| spread \ threshold | 0.60 | 0.63 | **0.65** | 0.68 | 0.70 |
|---|---|---|---|---|---|
| 0.0100 | −1.55 % | −0.99 % | −0.64 % | −1.31 % | −1.70 % |
| **0.0083 measured** | −1.11 % | −0.57 % | **−0.23 %** | −0.88 % | −1.42 % |
| 0.0060 | −0.60 % | −0.12 % | **+0.17 %** | −0.53 % | −1.13 % |
| 0.0030 | +0.09 % | +0.50 % | **+0.69 %** | −0.11 % | −0.76 % |

Threshold **0.65** is optimal at every spread: it halves the trade count and
cuts cost from 5.00 % to 3.84 %, because confident picks sit in higher-priced
tokens that are cheaper to trade. Above 0.70 the return collapses faster than
the cost falls. Break-even in spread sits near **0.007**.

## Execution is the lever, not the model

| variant | trades | cost | ROI |
|---|---|---|---|
| today: taker in, taker out | 7,560 | 3.61 % | −0.32 % |
| **+ maker exit** | 7,585 | 1.92 % | **+1.69 %** |
| + only books quoting ≤ 0.005 | 7,592 | 1.62 % | **+1.96 %** |
| pure maker | 7,669 | 0.00 % | +10.35 % ← **mirage** |

**Do not quote the pure-maker row.** Its extra 84 trades are tokens at a median
price of **0.0020** with returns to **+35,489 %**; at zero assumed cost they
stop being filtered as untradeable and drag the mean from 3.61 % to 10.35 %.
They are lottery tickets with no liquidity behind them.

**The price-band idea backfired**, and the reason is worth keeping: cost is
`(spread + legs·fee·min(p,1−p))/p`, which is *lowest* near p = 1, not in the
middle. Restricting to 0.20 ≤ p ≤ 0.80 pushed cost from 3.84 % to 6.26 %. The
model already trades at a median price of 0.89 — it exploits that structure
better than a hand-written band.

## The edge is thin and tail-carried

At the +1.69 % configuration:

| | |
|---|---|
| without the best 1 / 5 / 10 trades | +1.39 % / +0.91 % / +0.41 % |
| **without the best 50 of 7,585** | **−1.31 %** |
| median PnL per trade | **−0.16 %** |
| share of trades in profit | **44.3 %** |
| bootstrap 95 % CI | +0.64 % … +2.91 % |

The interval excludes zero, so this is not noise. But the median trade *loses*
and removing 0.7 % of trades flips the sign. Sharpe 0.07. That is a
lottery-shaped payoff, which makes sizing and fill assumptions matter more than
another feature — and points at the one modelling change with a mechanism
behind it: the model is trained on **direction** (`win` is binary at a 0.001
move) while the money is entirely in the **tail**.

## Adverse selection does not break the maker edge — slippage might

A resting limit fills disproportionately when the price moves against you, so
the +1.69 % had to be tested against that.

**Fill pattern alone does not break it.** Even never filling on a winner and
always filling on a loser leaves ROI at **+0.32 %**, above the −0.32 % taker
baseline. The saving — half the spread, one fee leg — applies to whichever
trades fill *regardless of outcome*. Adverse selection decides who gets the
discount, not whether it exists.

**What can break it is the price penalty for a missed fill**, which the first
model ignored: not filling means exiting later and worse.

| penalty on unfilled exits | × spread | ROI (winners fill) | ROI (only losers fill) |
|---|---|---|---|
| 0.0000 | 0× | +0.85 % | +0.32 % |
| 0.0050 | 0.6× | +0.57 % | −0.28 % |
| 0.0083 | 1.0× | +0.38 % | −0.67 % |
| 0.0150 | 1.8× | 0.00 % | −1.47 % |

Break-even against the taker baseline: **2.5× the spread** if winners fill,
**0.6×** in the hostile case.

---

# 4. What is running now

`collect_trades.py` and `paper_trades.py`, both under `runguard.sh`, measuring
the two numbers that cannot be recovered from history: **how often a resting
limit fills, and how much worse the exit is when it does not.**

`paper_trades.py` records the live book at decision time (bid, ask, spread,
depth both sides) and the price of a resting exit one full spread away, then
scans every print between entry and due. The fill rate it reports is an **upper
bound** — queue position is not modelled, so a print at the limit price may
have filled someone ahead. If the edge fails at that optimistic reading it was
never there.

Report at any time:

```bash
.venv/bin/python paper_trades.py --report --paper-db paper_trades.db
```

## Selecting the markets exposed a bigger constraint than the spread

The first selection sorted by lifetime volume and was wrong — only 1 of 19
markets had enough history to score. Lifetime volume is not current activity.

The corrected criterion (tight book *and* recently active) leaves **42 markets**
out of 8,767 trainable ones. And the cause is not a trade-off:

| spread band | markets | median active buckets / 7 days | share reaching 54 |
|---|---|---|---|
| ≤ 0.002 | 151 | 9 | 20 % |
| 0.002–0.005 | 111 | 13 | 11 % |
| 0.005–0.010 | 585 | 12 | 19 % |
| 0.010–0.030 | 490 | 9 | 11 % |
| > 0.030 | 348 | 7 | 9 % |

Correlation between log(spread) and log(activity): **−0.105**. Tight books are
not systematically thinner — **everything is thin**. The median market in every
band trades in 7–13 five-minute buckets *per week*.

At the median of the selected 42 — 127 active buckets per week, 1.5 per hour —
a market needs about **three days of collection** before it can be scored. The
first paper run skipped markets whose most recent bucket was 127 and 172
minutes old.

**This is a different constraint from everything above.** The backtest counts a
market trainable if it gathered 54 active buckets over its *lifetime*; live
scoring needs them inside a *moving window*. A strategy can look tradable on
history and still have almost no moments where it is actionable — so the
opportunity count, not the per-trade margin, may be what binds.

---

# 5. The data bridge: poly_data → Neuropoly

poly_data **collects**; this repo joins, filters and trains. `run_pipeline.sh`
runs with `PIPELINE_STAGES=collect` (markets + chain only), so
`processed/trades.csv` is never produced.

Two independent reasons, both measured. **Size**: orderFilled.csv is 163 GB
(625,627,683 events) and processed/trades.csv adds a 66-char hex `market_id`
plus an ISO timestamp per row, landing near 195 GB against 270 GB free — of
which 95 % would be discarded here. **Memory**:
`process_live._discover_missing_tokens()` runs before the chunked loop and
ignores `PROCESS_CHUNK_SIZE`, doing two `.unique().collect()` passes over the
whole file and then building Python sets of 77-char strings. It tripped the RAM
watchdog on 2026-09-16 and stopped collection for ten hours.

## The two-step bridge

1. **`build_registry.py`** — `markets.csv` → `data/market_registry.parquet`.
   Assigns a dense `Int32` `market_id` and carries the hex `condition_id`
   alongside. Re-run after poly_data appends markets; ids stay stable.
2. **`build_trades.py`** — `orderFilled.csv` → `data/trades.parquet`, joined
   against the registry with `how="inner"` so the universe filter applies
   *during* the scan. 625.6 M events → 33,042,239 rows and 0.81 GiB in
   **5.2 minutes**, a 200× reduction, because the excluded candle instruments
   are hyperactive.

`convert_to_parquet.py` targets poly_data **v1** and is superseded by both. Do
not repair it — it fails silently, writing an empty `trades.parquet`.

### Why a dense Int32 id

poly_data identifies markets by `condition_id` (66-char hex); this pipeline
casts `market_id` to `Int64` in eleven places. Both failure modes are **silent**:
`convert_to_parquet._build_token_lookup` skips every market and writes an empty
file; `segments.segment_of` catches the `ValueError` and labels all 3.4 M
markets `other`.

Ids are assigned to **every** market, not only kept ones, and a prior registry
is reloaded — so `market_id` survives a retuned filter and a grown export. The
universe is a *view* (`keep` / `exclude_reason`), never a subset.

### Two things measured rather than assumed

* **Gamma-backfill loss: 24,349 events of 625.6 M — 0.00 %.** Dropping
  poly_data's per-token Gamma lookup costs effectively nothing.
* **No dedup needed.** `convert_to_parquet.py` assumes "each trade appears
  TWICE"; for v2 that is false and would discard 87 % of the data. Every event
  carries exactly one USDC leg (1,736,203 maker-side + 263,797 taker-side =
  2,000,000 in the sample), and only 1.2 % of output rows repeat exactly —
  partial fills at one price in one transaction.

### markets.csv contains duplicates

poly_data's `update_markets` resume path dedups only against the last ~1500
rows, so a market re-fetched after a restart is appended again. Measured:
25,464 duplicates. The registry dedups on the input **and** on the prior
registry it joins against — a duplicated join partner *multiplies* rows — and
asserts uniqueness before writing.

---

# 6. Open questions

**Ranked by expected value, from the levers actually tested.**

1. **Train on magnitude, not direction.** `win` scores a 0.001 move the same as
   a 0.05 move, so the model optimises direction while the money is in the
   tail. A regression on `trade_return` targets what pays. This is the one
   modelling change with a mechanism behind it.
2. **Position sizing.** With a lottery-shaped payoff, sizing decides more than
   the signal. Kelly on a fat-tailed distribution is dangerous and is currently
   on by default.
3. **Fill probability** — being measured now; the largest unverified risk.
4. **Longer horizon (1 day).** D1 says `Ret>Cost` 47.9 % against 36.5 % at 8 h.
   Needs retraining.
5. **Use the depth** already recorded in `data/spreads.parquet` — a tight quote
   with five shares behind it is worthless.
6. **Re-pick the entry threshold on validation.** 0.65 was read off the *test*
   split; that is selection on the evaluation set and has to be corrected
   before any figure counts as out-of-sample.

**Exhausted:** price bands (backfire), more features or a bigger model
(direction already works), per-group models (closed by D0–D2).

**Still undecided:** weather markets are ~14 % of the universe and nobody has
said whether they belong. `will-sp500-close-above-6000` (prose) is kept while
`spx-above-6000-on-…` (templated) is excluded — the anchor is deliberate but the
asymmetry is real. And the `crypto` rule removes crypto-*themed* markets, not
only price bets, on purpose.
