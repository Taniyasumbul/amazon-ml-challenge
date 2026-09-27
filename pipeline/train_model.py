import sys, os, glob, time, gc
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, "/content/drive/MyDrive/amazon_norm/code")
from features import FEATS

NORM = "/content/drive/MyDrive/amazon_norm"
T0 = time.time()
def log(m): print(f"{m} | {time.time()-T0:.0f}s", flush=True)

# ---- Load features file by file, split by S1 entity (20% validation) ----
Xtr_l, ytr_l, Xva_l, yva_l, meta_l = [], [], [], [], []
files = sorted(glob.glob(f"{NORM}/feats/train_*.parquet"))
for f in files:
    country = os.path.basename(f).split("_")[1]
    d = pl.read_parquet(f).with_columns(
        ((pl.col("a").hash(seed=42) % 5) == 0).alias("is_val"))
    tr, va = d.filter(~pl.col("is_val")), d.filter(pl.col("is_val"))
    Xtr_l.append(tr.select(FEATS).to_numpy().astype(np.float32)); ytr_l.append(tr["y"].to_numpy())
    Xva_l.append(va.select(FEATS).to_numpy().astype(np.float32)); yva_l.append(va["y"].to_numpy())
    meta_l.append(va.select("a", "b", "y", "is_s3").with_columns(pl.lit(country).alias("country")))
    del d, tr, va; gc.collect()
Xtr = np.concatenate(Xtr_l); ytr = np.concatenate(ytr_l); del Xtr_l, ytr_l
Xva = np.concatenate(Xva_l); yva = np.concatenate(yva_l); del Xva_l, yva_l
meta = pl.concat(meta_l); del meta_l; gc.collect()
log(f"loaded: train={len(ytr):,} (pos {ytr.mean():.2%})  val={len(yva):,} (pos {yva.mean():.2%})")

# ---- Train ----
def checkpoint(env):
    if (env.iteration + 1) % 100 == 0:
        env.model.save_model(f"{NORM}/match_model_lgb.txt")
        print(f"  checkpoint saved at round {env.iteration+1}", flush=True)

dtr = lgb.Dataset(Xtr, ytr, feature_name=FEATS, free_raw_data=True)
dva = lgb.Dataset(Xva, yva, reference=dtr)
dtr.construct(); del Xtr; gc.collect()
params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              metric="average_precision", verbose=-1, num_threads=0)
m = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dva], valid_names=["val"],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50), checkpoint])
m.save_model(f"{NORM}/match_model_lgb.txt")
log(f"model saved, best iteration = {m.best_iteration}")

# ---- Validation predictions (needed for the decision step) ----
p = m.predict(Xva, num_iteration=m.best_iteration).astype(np.float32)
meta = meta.with_columns(pl.Series("p", p))
meta.write_parquet(f"{NORM}/val_preds.parquet")

print("\nTop 15 features by importance:")
for n, g in sorted(zip(FEATS, m.feature_importance("gain")), key=lambda x: -x[1])[:15]:
    print(f"  {n:14s} {g:,.0f}")

print("\nPair-level check (validation):")
for c in ["US", "India"]:
    v = meta.filter(pl.col("country") == c)
    for t in [0.3, 0.5, 0.7]:
        pred = v["p"] >= t
        tp = (pred & (v["y"] == 1)).sum()
        prec = tp / max(pred.sum(), 1); rec = tp / max((v["y"] == 1).sum(), 1)
        f05 = 1.25 * prec * rec / max(0.25 * prec + rec, 1e-9)
        print(f"  {c:6s} t={t}: precision={prec:.4f} recall={rec:.4f} F0.5={f05:.4f}")
log("ALL DONE")
