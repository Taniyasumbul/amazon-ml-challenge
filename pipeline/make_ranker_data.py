"""Labelled, uncapped blocking candidates for training the LambdaMART candidate ranker.

Usage:  PYTHONHASHSEED=0 python make_ranker_data.py <country> <src>
        e.g. US 2 | India 2 | India 3

For 40,000 sampled S1 of one country, blocks against the FULL train target pool,
keeps the top 300 candidates by bscore, labels each pair from the ground truth
and marks the first half of the sample as the ranker's fit set.
Output: $ER_WORK/ranker_data/<country>_S<src>.parquet (+ .json with the recall ceiling).
"""
import sys, os, json, time, gc, psutil
import polars as pl

SRC_DIR = os.environ.get("ER_SRC", "/content/business_entity_resolution/src")
NORM = os.environ.get("ER_WORK", "/content/drive/MyDrive/amazon_norm")
DATA = os.environ.get("ER_DATA", "/content/student_resource/student_resource/dataset")
sys.path.insert(0, SRC_DIR)
from blocking import token_df, candidates

OUTR = f"{NORM}/ranker_data"
os.makedirs(OUTR, exist_ok=True)
COLS = ["entity_id", "country", "name_core", "addr_toks", "addr_nums"]
BATCH = 10_000
NS1, TOPK, SEED = 40_000, 300, 7


def ram():
    return f"{psutil.Process().memory_info().rss / 1e9:.1f} GB"


def load(path, country):
    return (pl.scan_parquet(path).filter(pl.col("country") == country)
              .select(COLS).collect())


country, src = sys.argv[1], sys.argv[2]
path = f"{OUTR}/{country}_S{src}.parquet"
meta = f"{OUTR}/{country}_S{src}.json"
if os.path.exists(path) and os.path.exists(meta):
    print(f"{country} S{src}: already done, skip")
    sys.exit(0)

t0 = time.time()
s1 = load(f"{NORM}/train_s1.parquet", country)
tgt = load(f"{NORM}/train_s{src}.parquet", country)
uni = pl.concat([s1.select("name_core", "addr_toks"), tgt.select("name_core", "addr_toks")])
ndf = token_df(uni, "name_core", False)
adf = token_df(uni, "addr_toks", True)
del uni; gc.collect()

s1s = s1.sample(min(NS1, s1.height), seed=SEED).with_row_index("i1")
del s1; gc.collect()
tgt = tgt.with_row_index("it")

parts = []
for st in range(0, s1s.height, BATCH):
    ch = s1s.slice(st, BATCH).drop("i1").with_row_index("i1")
    c = candidates(ch, tgt, ndf, adf, TOPK, log=lambda *a: None)
    parts.append(c.with_columns(pl.col("i1") + st))
    print(f"  batch {st // BATCH}: {c.height:,} pairs | RAM {ram()}", flush=True)
    del ch, c; gc.collect()
cand = pl.concat(parts)
del parts, ndf, adf; gc.collect()

gt = pl.read_csv(f"{DATA}/train/train_ground_truth.tsv", separator="\t", infer_schema_length=0)
gt = gt.rename({"source1_entity_id": "a", "matched_entity_ids": "b"})
a_map = s1s.select("i1", pl.col("entity_id").alias("a"))
b_map = tgt.select("it", pl.col("entity_id").alias("b"))
truth = (gt.join(a_map, on="a", how="inner")
           .with_columns(pl.col("b").str.split(",")).explode("b")
           .with_columns(pl.col("b").str.strip_chars())
           .filter(pl.col("b").str.starts_with(f"S{src}-"))
           .join(b_map, on="b", how="inner")
           .select(pl.col("i1").cast(pl.Int32), pl.col("it").cast(pl.Int32)))
del gt, s1s, tgt, a_map, b_map; gc.collect()

pos = truth.with_columns(pl.lit(1, pl.Int8).alias("y"))
cand = cand.with_columns(pl.col("i1").cast(pl.Int32), pl.col("it").cast(pl.Int32))
cand = (cand.join(pos, on=["i1", "it"], how="left")
            .with_columns(pl.col("y").fill_null(0), (pl.col("i1") < NS1 // 2).alias("is_fit"))
            .drop("it"))

ceiling = cand["y"].sum() / truth.height
n_true_eval = truth.filter(pl.col("i1") >= NS1 // 2).height
cand.write_parquet(path)
with open(meta, "w") as f:
    json.dump({"n_true_eval": n_true_eval, "ceiling": ceiling}, f)
print(f"  DONE {cand.height / NS1:.0f}/S1, ceiling={ceiling:.4f}, {time.time() - t0:.0f}s | RAM {ram()}")
