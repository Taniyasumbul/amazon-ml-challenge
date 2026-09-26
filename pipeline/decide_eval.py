import json, polars as pl
NORM = "/content/drive/MyDrive/amazon_norm"
DATA = "/content/student_resource/student_resource/dataset"

vp = pl.read_parquet(f"{NORM}/val_preds.parquet")

# Validation S1 = train sample S1 in the 20% hash split (includes S1 with no candidates)
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
for c in ["US", "India"]:
    b = base.filter(pl.col("country") == c)
    print(f"{c}: {b.height:,} val S1, singletons={(b['n_true'] == 0).mean():.2%}")

def f05(sel):
    agg = sel.group_by("a").agg(pl.len().alias("n_pred"), pl.col("y").sum().alias("tp"))
    d = (base.join(agg, on="a", how="left")
             .with_columns(pl.col("n_pred").fill_null(0), pl.col("tp").fill_null(0)))
    prec = pl.col("tp") / pl.col("n_pred"); rec = pl.col("tp") / pl.col("n_true")
    f = (pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
           .when((pl.col("n_true") == 0) | (pl.col("n_pred") == 0) | (pl.col("tp") == 0)).then(0.0)
           .otherwise(1.25 * prec * rec / (0.25 * prec + rec)))
    r = d.with_columns(f.alias("f")).group_by("country").agg(pl.col("f").mean())
    r = dict(zip(r["country"], r["f"]))
    return r.get("US", 0), r.get("India", 0)

def cap(sel):
    return (sel.with_columns(pl.col("p").rank("ordinal", descending=True)
                             .over(["a", "is_s3"]).alias("rk"))
               .filter(pl.col("rk") <= pl.when(pl.col("is_s3") == 1).then(6).otherwise(5)))

us, ind = f05(vp.filter(pl.col("y") == 1))
print(f"\nCEILING (perfect decisions on our candidates): US={us:.4f} India={ind:.4f}\n")

vp1 = vp.filter(pl.col("p") >= pl.col("p").max().over("b"))   # one-parent rule
best = None
print(f"{'t':>5} | {'plain US':>9} {'plain IN':>9} | {'1-parent US':>11} {'1-parent IN':>11} {'avg':>7}")
for t in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    pu, pi = f05(cap(vp.filter(pl.col("p") >= t)))
    ou, oi = f05(cap(vp1.filter(pl.col("p") >= t)))
    avg = (ou + oi) / 2
    print(f"{t:5.2f} | {pu:9.4f} {pi:9.4f} | {ou:11.4f} {oi:11.4f} {avg:7.4f}")
    if best is None or avg > best[1]:
        best = (t, avg, ou, oi)

print(f"\nBEST: t={best[0]}  US={best[2]:.4f}  India={best[3]:.4f}  avg={best[1]:.4f}")
json.dump({"t": best[0], "one_parent": True, "caps": {"S2": 5, "S3": 6}},
          open(f"{NORM}/decision.json", "w"))
print("saved decision.json")
