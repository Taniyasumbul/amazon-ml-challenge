import sys, json, shutil, os, time
import polars as pl
sys.path.insert(0, "/content/drive/MyDrive/amazon_norm/code")
from decide import decide
NORM = "/content/drive/MyDrive/amazon_norm"
DATA = "/content/student_resource/student_resource/dataset"
T0 = time.time()

vp = pl.read_parquet(f"{NORM}/val_preds.parquet")
pos = vp.filter(pl.col("y") == 1).select("a", "b").with_columns(pl.lit(1).alias("y"))
d_all = vp.select("a", "b", "p", "is_s3")

ids = pl.read_parquet(f"{NORM}/train_block_ids.parquet").rename({"entity_id": "a"})
s1c = pl.read_parquet(f"{NORM}/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "a"})
val_a = ids.filter((pl.col("a").hash(seed=42) % 5) == 0).join(s1c, on="a")
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

results = []
for t2 in [0.6, 0.65, 0.7, 0.75, 0.8]:
    for t3 in [0.6, 0.65, 0.7, 0.75, 0.8]:
        cfg = {"mode": "threshold", "t2": t2, "t3": t3}
        results.append((cfg, *f05(decide(d_all, cfg))))
for t_low in [0.05, 0.1, 0.2, 0.3, 0.4]:
    for k0 in [0.8, 1.0, 1.25]:
        cfg = {"mode": "expf", "t_low": t_low, "k0": k0}
        results.append((cfg, *f05(decide(d_all, cfg))))
print(f"tested {len(results)} rules | {time.time()-T0:.0f}s\n")

baseline = next(r for r in results if r[0] == {"mode": "threshold", "t2": 0.7, "t3": 0.7})
print(f"BASELINE (current submission): US={baseline[1]:.4f} India={baseline[2]:.4f} avg={baseline[3]:.4f}\n")
results.sort(key=lambda r: -r[3])
print("TOP 8:")
for cfg, u, i, a in results[:8]:
    print(f"  avg={a:.4f}  US={u:.4f}  India={i:.4f}  {cfg}")

best = results[0][0]
if os.path.exists(f"{NORM}/decision.json") and not os.path.exists(f"{NORM}/decision_v1.json"):
    shutil.copy(f"{NORM}/decision.json", f"{NORM}/decision_v1.json")
json.dump(best, open(f"{NORM}/decision.json", "w"))
print(f"\nsaved best rule to decision.json: {best}  (gain vs baseline: {results[0][3]-baseline[3]:+.4f})")
