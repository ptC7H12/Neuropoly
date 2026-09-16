import sys; sys.path.insert(0,'/root/Neuropoly')
import polars as pl, lightgbm as lgb, numpy as np
from config import PipelineConfig
from pipeline.features import get_feature_columns
from pipeline.splitter import walk_forward_split
from pipeline.model import predict
R="/root/Neuropoly/"
cfg = PipelineConfig(); cfg.label.forward_window_buckets = 96
lf = pl.scan_parquet(R+"labeled.parquet"); fc = get_feature_columns(lf.collect_schema())
labeled = lf.filter(pl.col("win").is_not_null()).collect()
split = walk_forward_split(labeled, fc, cfg.split, cfg.label, cfg.bucket.bucket_minutes); del labeled
y = predict(lgb.Booster(model_file=R+"model_d2_8h.txt"), split.test_X)

SPREAD, FEE = 0.0083, 0.04
sel = y >= 0.65
p, r = split.test_price[sel], split.test_ret[sel]
mp = np.minimum(p, 1-p)
cost_tt = (SPREAD   + 2*FEE*mp)/p
cost_tm = (SPREAD/2 + 1*FEE*mp)/p
ok = cost_tt <= 0.5
p, r, cost_tt, cost_tm, mp = p[ok], r[ok], cost_tt[ok], cost_tm[ok], mp[ok]
n = len(r); rng = np.random.default_rng(0)
base_tt = float((r-cost_tt).mean())
print(f"Trades {n:,}   Basislinie (immer Taker raus): {base_tt:+.2%}\n")

# Nicht gefuellt = du steigst spaeter und schlechter aus.
# 'slip' = zusaetzlicher Renditeverlust auf genau diesen Trades, in Preiseinheiten,
# relativ zur Position (also /p wie alle anderen Kosten auch).
print("=== Maker-Ausstieg MIT Preisstrafe auf nicht gefuellten Trades ===")
print("  Fuellregel: nur guenstige Bewegungen fuellen (f_win=1, f_lose=0)")
print(f"  {'Strafe (Preiseinheiten)':<28}{'entspricht':>14}{'ROI':>10}")
print("  "+"-"*54)
filled = r > 0
for slip in (0.0, 0.002, 0.005, 0.0083, 0.015, 0.03):
    pen = np.where(filled, 0.0, slip/p)
    v = float((r - np.where(filled, cost_tm, cost_tt) - pen).mean())
    print(f"  {slip:<28.4f}{slip/SPREAD:>12.1f}x Spread{v:>10.2%}")

print("\n=== Worst case kombiniert: nur Verlierer fuellen UND Preisstrafe ===")
filled_adv = r <= 0
print(f"  {'Strafe':<28}{'ROI':>10}")
print("  "+"-"*40)
for slip in (0.0, 0.002, 0.005, 0.0083, 0.015):
    pen = np.where(filled_adv, 0.0, slip/p)
    v = float((r - np.where(filled_adv, cost_tm, cost_tt) - pen).mean())
    mark = "" if v > base_tt else "  <- schlechter als Taker"
    print(f"  {slip:<28.4f}{v:>10.2%}{mark}")

print("\n=== Ab welcher Strafe lohnt sich Maker nicht mehr? ===")
for rule, fmask in (("nur Gewinner fuellen", r>0), ("nur Verlierer fuellen", r<=0)):
    lo, hi = 0.0, 0.5
    for _ in range(40):
        mid = (lo+hi)/2
        pen = np.where(fmask, 0.0, mid/p)
        v = float((r - np.where(fmask, cost_tm, cost_tt) - pen).mean())
        if v > base_tt: lo = mid
        else: hi = mid
    print(f"  {rule:<24}: bis {lo:.4f} Preiseinheiten ({lo/SPREAD:.1f}x Spread)")
