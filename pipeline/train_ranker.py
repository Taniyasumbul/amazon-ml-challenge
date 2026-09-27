"""Train the LambdaMART candidate ranker used by block_run.py.

Usage:  python train_ranker.py
Input:  $ER_WORK/ranker_data/<country>_S<src>.parquet  (from make_ranker_data.py)
Output: $ER_WORK/block_ranker_lgb.txt, plus a recall@K report (bscore vs ranker)
        on the held-out half of each run.
"""
import os, json, gc
import numpy as np, polars as pl, lightgbm as lgb

NORM = os.environ.get("ER_WORK", "/content/drive/MyDrive/amazon_norm")
OUTR = f"{NORM}/ranker_data"
RUNS = [("US", "2"), ("India", "2"), ("India", "3")]

# Must stay identical to add_feats/FEATS in block_run.py.
FEATS = ([f"f{i}" for i in range(9)] + ["nfam", "bscore", "b_rel", "b_rank", "n_cand"]
         + [f"f{i}_rel" for i in range(9)])


def add_feats(d):
    rel = [(pl.col(f"f{i}") / (pl.col(f"f{i}").max().over("i1") + 1e-6)).alias(f"f{i}_rel")
           for i in range(9)]
    return d.with_columns(
        (pl.col("bscore") / pl.col("bscore").max().over("i1")).alias("b_rel"),
        pl.col("bscore").rank("ordinal", descending=True).over("i1").cast(pl.Float32).alias("b_rank"),
        pl.len().over("i1").cast(pl.Float32).alias("n_cand"),
        *rel)


def recall_at(df, col, k, n_true):
    r = pl.col(col).rank("ordinal", descending=True).over("i1")
    return df.with_columns(r.alias("r")).filter(pl.col("r") <= k)["y"].sum() / n_true


parts, evals = [], {}
for k, (c, s) in enumerate(RUNS):
    d = add_feats(pl.read_parquet(f"{OUTR}/{c}_S{s}.parquet"))
    with open(f"{OUTR}/{c}_S{s}.json") as f:
        n_true = json.load(f)["n_true_eval"]
    fit = d.filter(pl.col("is_fit"))
    fit = fit.filter(pl.col("y").max().over("i1") == 1)          # groups with at least one match
    parts.append(fit.with_columns((pl.col("i1") + k * 10_000_000).alias("gid")))
    evals[(c, s)] = (d.filter(~pl.col("is_fit")), n_true)
    del d, fit; gc.collect()

fit = pl.concat(parts).sort("gid")
del parts; gc.collect()
groups = fit.group_by("gid", maintain_order=True).len()["len"].to_numpy()
print(f"fit: {fit.height:,} pairs in {len(groups):,} groups")

rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=200, learning_rate=0.1,
                    num_leaves=63, min_child_samples=100,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    n_jobs=-1, verbose=-1)
rk.fit(fit.select(FEATS).to_numpy().astype(np.float32), fit["y"].to_numpy(), group=groups)
rk.booster_.save_model(f"{NORM}/block_ranker_lgb.txt")
del fit; gc.collect()
print("ranker saved\n")

for (c, s), (ev, n_true) in evals.items():
    score = rk.booster_.predict(ev.select(FEATS).to_numpy().astype(np.float32))
    ev = ev.with_columns(pl.Series("rk", score))
    for k in [30, 40, 50, 60]:
        print(f"{c} S{s}  K={k}  bscore={recall_at(ev, 'bscore', k, n_true):.4f}  "
              f"ranker={recall_at(ev, 'rk', k, n_true):.4f}")
    print()
