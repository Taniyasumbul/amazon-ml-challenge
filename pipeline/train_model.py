import os, glob, time, gc
import numpy as np, polars as pl, lightgbm as lgb
from features import FEATS
from validation import (EARLY_STOP_FOLD, HOLDOUT_FOLD, TUNE_FOLD,
                        fold_expr)

NORM = "/content/drive/MyDrive/amazon_norm"
T0 = time.time()
def log(m): print(f"{m} | {time.time()-T0:.0f}s", flush=True)


def stack(parts):
    parts = [part for part in parts if len(part)]
    if not parts:
        raise ValueError("validation split contains no candidate pairs")
    return np.concatenate(parts, axis=0)


# S1 IDs are assigned wholly to one group. The development model uses 70% for
# fitting and 10% for early stopping; tuning and final reporting each get 10%.
Xtr_l, ytr_l, Xes_l, yes_l, Xev_l, meta_l = [], [], [], [], [], []
files = sorted(glob.glob(f"{NORM}/feats/train_*.parquet"))
if not files:
    raise FileNotFoundError(f"no training feature files found under {NORM}/feats")

for f in files:
    country = os.path.basename(f).split("_")[1]
    d = pl.read_parquet(f).with_columns(fold_expr().alias("_fold"))
    tr = d.filter(pl.col("_fold") >= 3)
    es = d.filter(pl.col("_fold") == EARLY_STOP_FOLD)
    ev = d.filter(pl.col("_fold").is_in([TUNE_FOLD, HOLDOUT_FOLD]))
    if tr.height:
        Xtr_l.append(tr.select(FEATS).to_numpy().astype(np.float32)); ytr_l.append(tr["y"].to_numpy())
    if es.height:
        Xes_l.append(es.select(FEATS).to_numpy().astype(np.float32)); yes_l.append(es["y"].to_numpy())
    if ev.height:
        Xev_l.append(ev.select(FEATS).to_numpy().astype(np.float32))
        meta_l.append(
            ev.select("a", "b", "y", "is_s3", "_fold")
              .with_columns(
                  pl.lit(country).alias("country"),
                  pl.when(pl.col("_fold") == TUNE_FOLD)
                    .then(pl.lit("tune")).otherwise(pl.lit("holdout")).alias("split"))
              .drop("_fold"))
    del d, tr, es, ev
    gc.collect()

Xtr, ytr = stack(Xtr_l), np.concatenate(ytr_l)
Xes, yes = stack(Xes_l), np.concatenate(yes_l)
Xev = stack(Xev_l)
meta = pl.concat(meta_l)
del Xtr_l, ytr_l, Xes_l, yes_l, Xev_l, meta_l; gc.collect()
log(f"development data: train={len(ytr):,} (pos {ytr.mean():.2%})  "
    f"early-stop={len(yes):,} (pos {yes.mean():.2%})  eval={len(Xev):,}")

params = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              metric="average_precision", verbose=-1, num_threads=0)
dtr = lgb.Dataset(Xtr, ytr, feature_name=FEATS, free_raw_data=True)
des = lgb.Dataset(Xes, yes, reference=dtr)
dtr.construct(); del Xtr; gc.collect()
dev_model = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[des], valid_names=["early_stop"],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])
best_iteration = dev_model.best_iteration or dev_model.current_iteration()
log(f"development model best iteration = {best_iteration}")

# These predictions are made before the final all-data refit, so neither the
# tuning set nor the report holdout has influenced their model parameters.
p = dev_model.predict(Xev, num_iteration=best_iteration).astype(np.float32)
meta = meta.with_columns(pl.Series("p", p))
meta.write_parquet(f"{NORM}/val_preds.parquet")
log(f"saved grouped tune/holdout predictions: {meta.height:,} candidate pairs")

# Refit the deployment model on all labeled candidate pairs at the selected
# iteration count. Keep the untouched holdout predictions above as the report.
del dtr, des, dev_model, Xes, yes, Xev, ytr, meta, p; gc.collect()
Xfull_l, yfull_l = [], []
for f in files:
    d = pl.read_parquet(f)
    if d.height:
        Xfull_l.append(d.select(FEATS).to_numpy().astype(np.float32))
        yfull_l.append(d["y"].to_numpy())
    del d
    gc.collect()
Xfull, yfull = stack(Xfull_l), np.concatenate(yfull_l)
del Xfull_l, yfull_l; gc.collect()
log(f"refitting deployment model on all candidates: {len(yfull):,} pairs")
dfull = lgb.Dataset(Xfull, yfull, feature_name=FEATS, free_raw_data=True)
dfull.construct(); del Xfull, yfull; gc.collect()
model = lgb.train(params, dfull, num_boost_round=best_iteration)
model.save_model(f"{NORM}/match_model_lgb.txt")
log(f"saved deployment model: {best_iteration} iterations")

print("\nTop 15 deployment features by importance:")
for n, g in sorted(zip(FEATS, model.feature_importance("gain")), key=lambda x: -x[1])[:15]:
    print(f"  {n:14s} {g:,.0f}")
log("ALL DONE")
