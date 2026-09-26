import sys, os, glob, time, gc
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, "/content/drive/MyDrive/amazon_norm/code")
from features import FEATS

NORM = "/content/drive/MyDrive/amazon_norm"
T0 = time.time()
m = lgb.Booster(model_file=f"{NORM}/match_model_lgb.txt")

metas = []
for f in sorted(glob.glob(f"{NORM}/feats/train_*.parquet")):
    country = os.path.basename(f).split("_")[1]
    va = pl.read_parquet(f).filter((pl.col("a").hash(seed=42) % 5) == 0)   # same 20% split as training
    p = m.predict(va.select(FEATS).to_numpy().astype(np.float32)).astype(np.float32)
    metas.append(va.select("a", "b", "y", "is_s3")
                   .with_columns(pl.lit(country).alias("country"), pl.Series("p", p)))
    print(f"  {os.path.basename(f)}: {va.height:,} val pairs | {time.time()-T0:.0f}s", flush=True)
    del va; gc.collect()

meta = pl.concat(metas)
meta.write_parquet(f"{NORM}/val_preds.parquet")
print(f"saved val_preds.parquet: {meta.height:,} rows")

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
print(f"ALL DONE | {time.time()-T0:.0f}s")
