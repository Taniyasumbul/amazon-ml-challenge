import sys, time, gc
import numpy as np, polars as pl, lightgbm as lgb
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
sys.path.insert(0, "/content/drive/MyDrive/amazon_norm/code")
from features import prep, REC_COLS
from decide import decide

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
vp = add_ctx(pl.read_parquet(f"{NORM}/val_preds.parquet"))
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
pos = vp.filter(pl.col("y") == 1).select("a", "b").with_columns(pl.lit(1).alias("y"))

ids_all = pl.read_parquet(f"{NORM}/train_block_ids.parquet").rename({"entity_id": "a"})
s1c = pl.read_parquet(f"{NORM}/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "a"})
val_a = (ids_all.filter((pl.col("a").hash(seed=42) % 5) == 0)
                .filter((pl.col("a").hash(seed=7) % 2) == 1).join(s1c, on="a"))
gt = pl.read_csv(f"{DATA}/train/train_ground_truth.tsv", separator="\t", infer_schema_length=0)
n_true = (gt.rename({"source1_entity_id": "a", "matched_entity_ids": "b"})
            .join(val_a.select("a"), on="a", how="semi")
            .with_columns(pl.col("b").str.split(",")).explode("b")
            .with_columns(pl.col("b").str.strip_chars())
            .filter(pl.col("b").is_not_null() & (pl.col("b") != ""))
            .group_by("a").len().rename({"len": "n_true"}))
base = val_a.join(n_true, on="a", how="left").with_columns(pl.col("n_true").fill_null(0))

def f05(sel):
    sel = sel.join(pos, on=["a", "b"], how="left").with_columns(pl.col("y").fill_null(0))
    agg = sel.group_by("a").agg(pl.len().alias("n_pred"), pl.col("y").sum().alias("tp"))
    d = base.join(agg, on="a", how="left").with_columns(pl.col("n_pred").fill_null(0), pl.col("tp").fill_null(0))
    prec = pl.col("tp") / pl.col("n_pred"); rec = pl.col("tp") / pl.col("n_true")
    f = (pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
           .when((pl.col("n_true") == 0) | (pl.col("n_pred") == 0) | (pl.col("tp") == 0)).then(0.0)
           .otherwise(1.25 * prec * rec / (0.25 * prec + rec)))
    r = d.with_columns(f.alias("f")).group_by("country").agg(pl.col("f").mean())
    r = dict(zip(r["country"], r["f"]))
    return r["US"], r["India"], (r["US"] + r["India"]) / 2

res = {}
for name, col in [("stage 1 (current)", "p"), ("stage 2 (new)", "p2")]:
    best = None
    for t in [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
        s = f05(decide(ev.select("a", "b", pl.col(col).alias("p"), "is_s3"),
                       {"mode": "threshold", "t2": t, "t3": t}))
        if best is None or s[2] > best[1][2]:
            best = (t, s)
    res[name] = best
    print(f"{name:18s}: best t={best[0]}  US={best[1][0]:.4f}  India={best[1][1]:.4f}  avg={best[1][2]:.4f}")
gain = res["stage 2 (new)"][1][2] - res["stage 1 (current)"][1][2]
print(f"\nGAIN from stage 2: {gain:+.4f}")
log("ALL DONE")
