"""Blocking + learned re-ranking + top-K candidate selection.

Usage:  PYTHONHASHSEED=0 python block_run.py <split> <country> <src> <K> [s1_ids.parquet]
        e.g. test India 2 40   |   train US 3 30 $ER_WORK/train_block_ids.parquet

Target-side blocking keys are built once per (country, source); S1 is processed in
chunks of 25,000. For each S1 the top 300 targets by summed per-family IDF evidence
are re-ranked by the LambdaMART ranker (block_ranker_lgb.txt) and the top K are kept.
Output: $ER_WORK/blocks/<split>_<country>_S<src>_<chunk>.parquet with columns
a, b, f0..f8, nfam, bscore, b_rel, rscore, rrank (consumed by features.py).
"""
import sys, os, time, gc, psutil
import numpy as np, polars as pl, lightgbm as lgb

SRC_DIR = os.environ.get("ER_SRC", "/content/business_entity_resolution/src")
NORM = os.environ.get("ER_WORK", "/content/drive/MyDrive/amazon_norm")
sys.path.insert(0, SRC_DIR)
import blocking
from blocking import token_df, keys_frame
from keys import N_FAM
blocking.KEY_CHUNK = 100_000          # smaller pieces -> lower peak RAM

OUT = f"{NORM}/blocks"
os.makedirs(OUT, exist_ok=True)
COLS = ["entity_id", "country", "name_core", "addr_toks", "addr_nums"]
TOPK_RAW = 300        # same as ranker training
CHUNK = 25_000        # S1 records per chunk
MAX_KEY_DF = 120
PIECE = 500_000       # target records per key-building piece

split, country, src, K = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ids_file = sys.argv[5] if len(sys.argv) > 5 else None
T0 = time.time()


def log(msg):
    rss = psutil.Process().memory_info().rss / 1e9
    print(f"{msg} | {time.time()-T0:.0f}s | RAM {rss:.1f} GB", flush=True)


def load(path):
    return (pl.scan_parquet(path).filter(pl.col("country") == country)
              .select(COLS).collect())


def cpath(k):
    return f"{OUT}/{split}_{country}_S{src}_{k:03d}.parquet"


# ---- S1 and which chunks are still to do ----
s1_all = load(f"{NORM}/{split}_s1.parquet")
s1 = s1_all
if ids_file:
    s1 = s1_all.join(pl.read_parquet(ids_file), on="entity_id", how="semi")
n_chunks = (s1.height + CHUNK - 1) // CHUNK
todo = [k for k in range(n_chunks) if not os.path.exists(cpath(k))]
print(f"{split} {country} S{src} K={K}: S1={s1.height:,} chunks={n_chunks} todo={len(todo)}", flush=True)
if not todo:
    sys.exit(0)

# ---- Token document frequencies over full country S1 + target ----
tgt = load(f"{NORM}/{split}_s{src}.parquet")
uni = pl.concat([s1_all.select("name_core", "addr_toks"),
                 tgt.select("name_core", "addr_toks")])
ndf = token_df(uni, "name_core", False)
adf = token_df(uni, "addr_toks", True)
del uni, s1_all; gc.collect()
log(f"  df tables built (tgt={tgt.height:,})")

# ---- Target keys: built ONCE ----
ntgt = tgt.height
tgt_ids = pl.DataFrame({"it": np.arange(ntgt, dtype=np.int32), "b": tgt["entity_id"]})
parts = []
for st in range(0, ntgt, PIECE):
    kf = keys_frame(tgt.slice(st, PIECE), ndf, adf, "t")
    parts.append(kf.with_columns((pl.col("it") + st).cast(pl.Int32)))
    del kf; gc.collect()
del tgt; gc.collect()
kt = pl.concat(parts); del parts; gc.collect()
kdf = kt.group_by("h").len().filter(pl.col("len") <= MAX_KEY_DF)
lg = float(np.log(max(ntgt, 2)))
kt = (kt.join(kdf, on="h", how="inner")
        .with_columns((lg - pl.col("len").cast(pl.Float32).log()).cast(pl.Float32).alias("w"))
        .drop("len"))
keep_h = kdf.select("h"); del kdf; gc.collect()
log(f"  target keys built: {kt.height:,}")

# ---- Ranker (same features as train_ranker.py) ----
booster = lgb.Booster(model_file=f"{NORM}/block_ranker_lgb.txt")
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


# ---- S1 chunks ----
for k in todo:
    ch = s1.slice(k * CHUNK, CHUNK)
    k1 = keys_frame(ch, ndf, adf, "1").join(keep_h, on="h", how="semi")
    j = k1.join(kt, on="h", how="inner")
    del k1
    agg = j.group_by(["i1", "it"]).agg(
        [pl.when(pl.col("famt") == f).then(pl.col("w")).otherwise(0.0)
           .sum().cast(pl.Float32).alias(f"f{f}") for f in range(N_FAM)]
        + [pl.col("famt").n_unique().cast(pl.UInt8).alias("nfam")])
    del j; gc.collect()
    agg = agg.with_columns(pl.sum_horizontal([f"f{f}" for f in range(N_FAM)])
                             .cast(pl.Float32).alias("bscore"))
    agg = (agg.sort(["i1", "bscore"], descending=[False, True])
              .group_by("i1", maintain_order=True).head(TOPK_RAW))
    if agg.height == 0:
        pl.DataFrame(schema={"a": pl.Utf8, "b": pl.Utf8}).write_parquet(cpath(k))
        log(f"  chunk {k}: empty"); continue
    agg = add_feats(agg)
    score = booster.predict(agg.select(FEATS).to_numpy().astype(np.float32))
    agg = agg.with_columns(pl.Series("rscore", score, dtype=pl.Float32))
    agg = agg.with_columns(pl.col("rscore").rank("ordinal", descending=True)
                             .over("i1").alias("rrank")).filter(pl.col("rrank") <= K)
    a_map = (ch.select(pl.col("entity_id").alias("a")).with_row_index("i1")
               .with_columns(pl.col("i1").cast(pl.Int32)))
    agg = (agg.with_columns(pl.col("i1").cast(pl.Int32), pl.col("it").cast(pl.Int32))
              .join(a_map, on="i1").join(tgt_ids, on="it")
              .select(["a", "b"] + [f"f{i}" for i in range(9)]
                      + ["nfam", "bscore", "b_rel", "rscore", "rrank"]))
    agg.write_parquet(cpath(k))
    log(f"  chunk {k}: {ch.height:,} S1 -> {agg.height:,} pairs")
    del agg, ch; gc.collect()

log("  ALL DONE")
