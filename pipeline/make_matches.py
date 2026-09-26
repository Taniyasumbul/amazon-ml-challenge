import os, glob, json, time, gc
import polars as pl

NORM = "/content/drive/MyDrive/amazon_norm"
T0 = time.time()
dec = json.load(open(f"{NORM}/decision.json"))
t = dec["t"]
print(f"decision: t={t}, caps={dec['caps']}", flush=True)

s1 = pl.read_parquet(f"{NORM}/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "a"})
parts = []
for c in ["US", "India", "France"]:
    sel_c = []
    for s in ["2", "3"]:
        d = pl.read_parquet(sorted(glob.glob(f"{NORM}/preds/test_{c}_S{s}_*.parquet")), columns=["a", "b", "p"])
        sel = d.filter((pl.col("p") >= pl.col("p").max().over("b")) & (pl.col("p") >= t))
        k = dec["caps"]["S3"] if s == "3" else dec["caps"]["S2"]
        sel = (sel.with_columns(pl.col("p").rank("ordinal", descending=True).over("a").alias("rk"))
                  .filter(pl.col("rk") <= k).select("a", "b"))
        sel_c.append(sel)
        print(f"  {c} S{s}: {sel.height:,} matches | {time.time()-T0:.0f}s", flush=True)
        del d; gc.collect()
    a_c = s1.filter(pl.col("country") == c).select("a")
    allsel = pl.concat(sel_c)
    m = allsel.group_by("a").agg(pl.col("b").str.join(",").alias("matched_entity_ids"))
    m = a_c.join(m, on="a", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
    print(f"{c}: with >=1 match={(m['matched_entity_ids'] != '').mean():.2%}  "
          f"avg matches/S1={allsel.height / a_c.height:.2f}", flush=True)
    parts.append(m)
    del sel_c, allsel; gc.collect()

match = pl.concat(parts).rename({"a": "source1_entity_id"})
assert match.height == s1.height == match["source1_entity_id"].n_unique(), "row count mismatch"
match.write_csv(f"{NORM}/output/matching_results.tsv", separator="\t", quote_style="never")
print(f"\nwrote {match.height:,} rows to matching_results.tsv | {time.time()-T0:.0f}s")
