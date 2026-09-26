import time, gc
import numpy as np, polars as pl, lightgbm as lgb
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from features import prep, REC_COLS
from decide import decide
from validation import TUNE_FOLD, score_summary
from validation_data import load_validation_base

NORM = "/content/drive/MyDrive/amazon_norm"
DATA = "/content/student_resource/student_resource/dataset"
T0 = time.time()
def log(m): print(f"{m} | {time.time()-T0:.0f}s", flush=True)

ANC = ["anc_n_ratio", "anc_n_tset", "anc_n_jw", "anc_a_tset", "anc_n0_eq", "anc_n"]
F2 = ["p", "is_s3", "p_rank", "p_gap", "p_sum", "n_conf"] + ANC

def add_ctx(d):
    return d.with_columns(
        pl.col("p").rank("ordinal", descending=True).over("a").cast(pl.Float32).alias("p_rank"),
        (pl.col("p").max().over("a") - pl.col("p")).alias("p_gap"),
        pl.col("p").sum().over("a").alias("p_sum"),
        ((pl.col("p") >= 0.9).sum().over("a") - (pl.col("p") >= 0.9).cast(pl.UInt32)).cast(pl.Float32).alias("n_conf"))

def anchor_feats(R, anchors, recs):
    X = R.select("a", "b").join(anchors, on="a").filter(pl.col("b") != pl.col("b2"))
    r1 = recs.select(pl.col("id_r").alias("b"), pl.col("nc_r").alias("nc1"), pl.col("ad_r").alias("ad1"), pl.col("n0_r").alias("n01"))
    r2 = recs.select(pl.col("id_r").alias("b2"), pl.col("nc_r").alias("nc2"), pl.col("ad_r").alias("ad2"), pl.col("n0_r").alias("n02"))
    X = X.join(r1, on="b", how="left").join(r2, on="b2", how="left")
    def cp(x, y, sc):
        return cpdist(X[x].fill_null("").to_list(), X[y].fill_null("").to_list(), scorer=sc, workers=-1, dtype=np.float32)
    X = X.with_columns(pl.Series("s_nr", cp("nc1", "nc2", fuzz.ratio)),
                       pl.Series("s_nt", cp("nc1", "nc2", fuzz.token_set_ratio)),
                       pl.Series("s_jw", cp("nc1", "nc2", JaroWinkler.normalized_similarity)),
                       pl.Series("s_at", cp("ad1", "ad2", fuzz.token_set_ratio)),
                       (pl.col("n01") == pl.col("n02")).fill_null(False).cast(pl.Float32).alias("s_n0"))
    return X.group_by(["a", "b"]).agg(
        pl.col("s_nr").max().alias("anc_n_ratio"), pl.col("s_nt").max().alias("anc_n_tset"),
        pl.col("s_jw").max().alias("anc_n_jw"), pl.col("s_at").max().alias("anc_a_tset"),
        pl.col("s_n0").max().alias("anc_n0_eq"), pl.len().cast(pl.Float32).alias("anc_n"))

# ---- Build stage-2 data from validation predictions ----
vp = add_ctx(pl.read_parquet(f"{NORM}/val_preds.parquet")
               .filter(pl.col("split") == "tune"))
R = vp.filter(pl.col("p") >= 0.05)                               # only uncertain/likely pairs are rescored
anchors = vp.filter(pl.col("p") >= 0.9).select("a", pl.col("b").alias("b2"))
ids = pl.concat([R.select(pl.col("b").alias("entity_id")),
                 anchors.select(pl.col("b2").alias("entity_id"))]).unique()
raw = pl.concat([pl.scan_parquet(f"{NORM}/train_s{s}.parquet").select(REC_COLS)
                   .join(ids.lazy(), on="entity_id", how="semi").collect() for s in "23"])
recs = prep(raw, "r"); del raw; gc.collect()
log(f"rescoring {R.height:,} pairs, anchors={anchors.height:,}")

AF = anchor_feats(R, anchors, recs)
R = R.join(AF, on=["a", "b"], how="left").with_columns([pl.col(c).fill_null(-1.0) for c in ANC])
R = R.with_columns(((pl.col("a").hash(seed=7) % 2) == 0).alias("h_tr"))
log("anchor features done")

tr = R.filter(pl.col("h_tr"))
m2 = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                    feature_fraction=0.9, verbose=-1),
               lgb.Dataset(tr.select(F2).to_numpy().astype(np.float32), tr["y"].to_numpy()),
               num_boost_round=300)
m2.save_model(f"{NORM}/stage2_model.txt")
R = R.with_columns(pl.Series("p2", m2.predict(R.select(F2).to_numpy().astype(np.float32)).astype(np.float32)))
log("stage-2 model trained")

# ---- Evaluate on the OTHER half of validation businesses ----
vp2 = (vp.join(R.select("a", "b", "p2"), on=["a", "b"], how="left")
         .with_columns(pl.col("p2").fill_null(pl.col("p"))))
ev = vp2.filter((pl.col("a").hash(seed=7) % 2) == 1)
base_all, _ = load_validation_base(NORM, DATA, TUNE_FOLD)
base = base_all.filter((pl.col("a").hash(seed=7) % 2) == 1)

def f05(sel):
    labels = vp.select("a", "b", "y")
    selected = (sel.join(labels, on=["a", "b"], how="left")
                  .with_columns(pl.col("y").fill_null(0)))
    scores = score_summary(base, selected)
    return scores["overall"], scores["country"]

res = {}
for name, col in [("stage 1 (current)", "p"), ("stage 2 (new)", "p2")]:
    best = None
    for t in [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
        s = f05(decide(ev.select("a", "b", pl.col(col).alias("p"), "is_s3"),
                       {"mode": "threshold", "t2": t, "t3": t}))
        if best is None or s[0] > best[1][0]:
            best = (t, s)
    res[name] = best
    print(f"{name:18s}: best t={best[0]}  overall={best[1][0]:.4f}  country={best[1][1]}")
gain = res["stage 2 (new)"][1][0] - res["stage 1 (current)"][1][0]
print(f"\nGAIN from stage 2: {gain:+.4f}")
log("ALL DONE")
