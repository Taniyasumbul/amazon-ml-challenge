import sys, os, glob, time, gc, psutil
import numpy as np, polars as pl, lightgbm as lgb
from features import prep, pair_features, REC_COLS, FEATS

NORM = "/content/drive/MyDrive/amazon_norm"
OUT = f"{NORM}/preds"
os.makedirs(OUT, exist_ok=True)
country, src = sys.argv[1], sys.argv[2]
T0 = time.time()

def log(msg):
    rss = psutil.Process().memory_info().rss / 1e9
    print(f"{msg} | {time.time()-T0:.0f}s | RAM {rss:.1f} GB", flush=True)

files = sorted(glob.glob(f"{NORM}/blocks/test_{country}_S{src}_*.parquet"))
todo = [f for f in files if not os.path.exists(f"{OUT}/{os.path.basename(f)}")]
print(f"test {country} S{src}: files={len(files)} todo={len(todo)}", flush=True)
if not todo:
    sys.exit(0)

def load(path):
    return (pl.scan_parquet(path).filter(pl.col("country") == country)
              .select(REC_COLS).collect())

A = prep(load(f"{NORM}/test_s1.parquet"), "a")
B = prep(load(f"{NORM}/test_s{src}.parquet"), "b")
m = lgb.Booster(model_file=f"{NORM}/match_model_lgb.txt")
log("  records prepared, model loaded")

for f in todo:
    out = f"{OUT}/{os.path.basename(f)}"
    p = pl.read_parquet(f)
    if p.height == 0:
        pl.DataFrame(schema={"a": pl.Utf8, "b": pl.Utf8, "p": pl.Float32,
                             "is_s3": pl.Int8}).write_parquet(out)
        log(f"  {os.path.basename(f)}: empty"); continue
    d = pair_features(p, A, B, src == "3")
    prob = m.predict(d.select(FEATS).to_numpy().astype(np.float32)).astype(np.float32)
    (d.select("a", "b")
      .with_columns(pl.Series("p", prob), pl.lit(1 if src == "3" else 0, pl.Int8).alias("is_s3"))
      .write_parquet(out))
    log(f"  {os.path.basename(f)}: {d.height:,} pairs, mean p={prob.mean():.3f}")
    del p, d, prob; gc.collect()
log("  ALL DONE")
