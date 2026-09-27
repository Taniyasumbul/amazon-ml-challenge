"""Competition features: how strongly S1 `a` claims record `b` compared with rival S1s.
Usage: python comp_feats.py <split> <country> <src>
Train rivals come from blocks_all/ (all train S1); test rivals from blocks/ (all test S1).
Output is aligned with blocks/ chunk files: comp/<same file name>.
"""
import sys, os, glob, time, gc
import polars as pl

NORM = os.environ.get("ER_WORK", "/content/drive/MyDrive/amazon_norm")
split, country, src = sys.argv[1], sys.argv[2], sys.argv[3]
SRC_DIR = f"{NORM}/blocks_all" if split == "train" else f"{NORM}/blocks"
OUT = f"{NORM}/comp"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()

targets = sorted(glob.glob(f"{NORM}/blocks/{split}_{country}_S{src}_*.parquet"))
todo = [f for f in targets if not os.path.exists(f"{OUT}/{os.path.basename(f)}")]
print(f"{split} {country} S{src}: chunks={len(targets)} todo={len(todo)}", flush=True)
if not todo:
    sys.exit(0)

# records we need rival information for
need_b = (pl.concat([pl.read_parquet(f, columns=["b"]) for f in targets])
            .select(pl.col("b").hash().alias("bh")).unique())

# all S1 claiming those records (IDs hashed to integers to save memory)
d = (pl.scan_parquet(sorted(set(glob.glob(f"{SRC_DIR}/{split}_{country}_S{src}_*.parquet")) | set(targets)))
       .select(pl.col("a").hash().alias("ah"), pl.col("b").hash().alias("bh"), "rscore", "bscore")
       .unique(["ah", "bh"]).join(need_b.lazy(), on="bh", how="semi")
       .collect())
del need_b; gc.collect()

nc = pl.col("name_core").fill_null("")
names = (pl.scan_parquet(f"{NORM}/{split}_s1.parquet").filter(pl.col("country") == country)
           .select(pl.col("entity_id").hash().alias("ah"),
                   pl.when(nc == "").then(None).otherwise(nc.hash()).alias("nh"))
           .collect())
d = d.join(names, on="ah", how="left")
del names; gc.collect()

d = d.with_columns(pl.len().over("bh").alias("c_n"),
                   pl.col("rscore").rank("ordinal", descending=True).over("bh").alias("c_rank"),
                   pl.col("bscore").rank("ordinal", descending=True).over("bh").alias("c_rank_b"))
d = d.with_columns(
    pl.col("rscore").max().over("bh").alias("_r1"),
    pl.col("rscore").filter(pl.col("c_rank") == 2).max().over("bh").alias("_r2"),
    pl.col("bscore").max().over("bh").alias("_b1"),
    pl.col("bscore").filter(pl.col("c_rank_b") == 2).max().over("bh").alias("_b2"),
    pl.when(pl.col("nh").is_null()).then(0).otherwise(pl.len().over(["bh", "nh"]) - 1).alias("c_same"))
d = d.with_columns(
    (pl.col("rscore") - pl.when(pl.col("c_rank") == 1).then(pl.col("_r2")).otherwise(pl.col("_r1")))
      .fill_null(10.0).alias("c_margin_r"),
    (pl.col("bscore") - pl.when(pl.col("c_rank_b") == 1).then(pl.col("_b2")).otherwise(pl.col("_b1")))
      .fill_null(50.0).alias("c_margin_b"))
d = d.select("ah", "bh", *[pl.col(c).cast(pl.Float32) for c in ["c_n", "c_rank", "c_margin_r", "c_margin_b", "c_same"]])
gc.collect()
print(f"  rival table: {d.height:,} pairs | {time.time()-T0:.0f}s", flush=True)

for f in todo:
    keys = pl.read_parquet(f, columns=["a", "b"]).with_columns(
        pl.col("a").hash().alias("ah"), pl.col("b").hash().alias("bh"))
    out = keys.join(d, on=["ah", "bh"], how="left").drop("ah", "bh")
    out.write_parquet(f"{OUT}/{os.path.basename(f)}")
    print(f"  {os.path.basename(f)}: {out.height:,} pairs, found in rival table: "
          f"{out['c_n'].is_not_null().mean():.2%} | {time.time()-T0:.0f}s", flush=True)
print("  ALL DONE", flush=True)
