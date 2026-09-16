import sys; sys.path.insert(0,'/root/Neuropoly')
import polars as pl, lightgbm as lgb, numpy as np
from config import PipelineConfig
from pipeline.features import get_feature_columns
from pipeline.splitter import walk_forward_split
from pipeline.model import predict
from pipeline.evaluation import backtest
R="/root/Neuropoly/"
cfg = PipelineConfig(); cfg.label.forward_window_buckets = 96
lf = pl.scan_parquet(R+"labeled.parquet"); fc = get_feature_columns(lf.collect_schema())
labeled = lf.filter(pl.col("win").is_not_null()).collect()
split = walk_forward_split(labeled, fc, cfg.split, cfg.label, cfg.bucket.bucket_minutes); del labeled
booster = lgb.Booster(model_file=R+"model_d2_8h.txt")
y = predict(booster, split.test_X)
print(f"Test-Zeilen: {len(split.test_y):,}\n")

def bt(spread, legs, thr, mask=None):
    c = type(cfg.backtest.cost)(**{**cfg.backtest.cost.__dict__,
                                   "spread_abs": spread, "fee_legs": legs})
    m = slice(None) if mask is None else mask
    return backtest(split.test_y[m], y[m], trade_returns=split.test_ret[m],
                    entry_prices=split.test_price[m], cost=c, entry_threshold=thr,
                    fee_rate=cfg.backtest.fee_rate, max_position_usd=cfg.backtest.max_position_usd,
                    kelly_sizing=cfg.backtest.kelly_sizing, kelly_cap=cfg.backtest.kelly_cap,
                    initial_bankroll=cfg.backtest.initial_bankroll)

print("=== HEBEL 1: Maker statt Taker ===")
print(f"  {'Ausfuehrung':<34}{'Trades':>8}{'Rendite':>9}{'Kosten':>9}{'ROI':>9}")
print("  "+"-"*69)
for lbl, sp, legs in [
    ("Taker rein + Taker raus (heute)", 0.0083, 2),
    ("Taker rein + Maker raus",         0.00415, 1),
    ("Maker rein + Taker raus",         0.00415, 1),
    ("Maker rein + Maker raus",         0.0,     0),
]:
    b = bt(sp, legs, 0.65)
    print(f"  {lbl:<34}{b.total_trades:>8,}{b.mean_trade_return:>8.2%}{b.mean_cost:>9.2%}{b.roi:>9.2%}")

print("\n=== HEBEL 2: Preisband (Kosten skalieren mit 1/p) ===")
p = split.test_price
print(f"  {'Band':<22}{'Zeilen':>9}{'Trades':>8}{'Rendite':>9}{'Kosten':>9}{'ROI':>9}")
print("  "+"-"*66)
for lbl, lo, hi in [("alle",0.0,1.0),("p >= 0.10",0.10,1.0),("p >= 0.20",0.20,1.0),
                    ("0.20 <= p <= 0.80",0.20,0.80),("0.30 <= p <= 0.70",0.30,0.70)]:
    mask = (p >= lo) & (p <= hi)
    if mask.sum() < 100: continue
    b = bt(0.0083, 2, 0.65, mask)
    print(f"  {lbl:<22}{int(mask.sum()):>9,}{b.total_trades:>8,}{b.mean_trade_return:>8.2%}{b.mean_cost:>9.2%}{b.roi:>9.2%}")

print("\n=== HEBEL 3: beides kombiniert (Schwelle 0.65) ===")
print(f"  {'Variante':<40}{'Trades':>8}{'Kosten':>9}{'ROI':>9}")
print("  "+"-"*66)
band = (p >= 0.20) & (p <= 0.80)
for lbl, sp, legs, mk in [
    ("heute", 0.0083, 2, None),
    ("+ Preisband 0.20-0.80", 0.0083, 2, band),
    ("+ enge Maerkte (Spread 0.005)", 0.0050, 2, band),
    ("+ Maker-Ausstieg", 0.0025, 1, band),
    ("+ reiner Maker", 0.0, 0, band),
]:
    b = bt(sp, legs, 0.65, mk)
    print(f"  {lbl:<40}{b.total_trades:>8,}{b.mean_cost:>9.2%}{b.roi:>9.2%}")
