import sys, os, glob, time, gc, psutil
import polars as pl
from features import prep, pair_features, REC_COLS

NORM = "/content/drive/MyDrive/amazon_norm"
DATA = "/content/student_resource/student_resource/dataset"
OUTF = f"{NORM}/feats"
os.makedirs(OUTF, exist_ok=True)
country, src = sys.argv[1], sys.argv[2]
T0 = time.time()

def log(msg):
    rss = psutil.Process().memory_info().rss / 1e9
    print(f"{msg} | {time.time()-T0:.0f}s | RAM {rss:.1f} GB", flush=True)

files = sorted(glob.glob(f"{NORM}/blocks/train_{country}_S{src}_*.parquet"))
todo = [f for f in files if not os.path.exists(f"{OUTF}/{os.path.basename(f)}")]
print(f"train {country} S{src}: files={len(files)} todo={len(todo)}", flush=True)
if not todo:
    sys.exit(0)

def load(path):
    return (pl.scan_parquet(path).filter(pl.col("country") == country)
              .select(REC_COLS).collect())

A = prep(load(f"{NORM}/train_s1.parquet"), "a")
B = prep(load(f"{NORM}/train_s{src}.parquet"), "b")
log("  records prepared")

gt = pl.read_csv(f"{DATA}/train/train_ground_truth.tsv", separator="\t", infer_schema_length=0)
truth = (gt.rename({"source1_entity_id": "a", "matched_entity_ids": "b"})
           .with_columns(pl.col("b").str.split(",")).explode("b")
           .with_columns(pl.col("b").str.strip_chars())
           .filter(pl.col("b").str.starts_with(f"S{src}-"))
           .with_columns(pl.lit(1, pl.Int8).alias("y")))
del gt; gc.collect()

for f in todo:
    p = pl.read_parquet(f)
    d = pair_features(p, A, B, src == "3")
    d = d.join(truth, on=["a", "b"], how="left").with_columns(pl.col("y").fill_null(0))
    d.write_parquet(f"{OUTF}/{os.path.basename(f)}")
    log(f"  {os.path.basename(f)}: {d.height:,} pairs, positives={d['y'].sum():,}")
    del p, d; gc.collect()
log("  ALL DONE")
