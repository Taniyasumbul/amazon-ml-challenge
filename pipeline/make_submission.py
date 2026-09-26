import os, glob, json, time, gc
import polars as pl

NORM = "/content/drive/MyDrive/amazon_norm"
OUT = f"{NORM}/output"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
dec = json.load(open(f"{NORM}/decision.json"))
t = dec["t"]
print(f"decision: t={t}, one_parent={dec['one_parent']}, caps={dec['caps']}", flush=True)

s1 = pl.read_parquet(f"{NORM}/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "a"})
match_parts, cand_parts = [], []

for c in ["US", "India", "France"]:
    sel_c, cand_c = [], []
    for s in ["2", "3"]:
        files = sorted(glob.glob(f"{NORM}/preds/test_{c}_S{s}_*.parquet"))
        d = pl.read_parquet(files)
        cand_c.append(d.select("a", "b"))
        sel = d.filter(pl.col("p") >= pl.col("p").max().over("b"))     # one-parent rule
        sel = sel.filter(pl.col("p") >= t)                                 # cutoff
        k = dec["caps"]["S3"] if s == "3" else dec["caps"]["S2"]
        sel = (sel.with_columns(pl.col("p").rank("ordinal", descending=True).over("a").alias("rk"))
                  .filter(pl.col("rk") <= k))                              # caps
        sel_c.append(sel.select("a", "b"))
        print(f"  {c} S{s}: {d.height:,} candidates -> {sel.height:,} matches "
              f"| {time.time()-T0:.0f}s", flush=True)
        del d, sel; gc.collect()

    a_c = s1.filter(pl.col("country") == c).select("a")
    m = pl.concat(sel_c).group_by("a").agg(pl.col("b").str.join(",").alias("matched_entity_ids"))
    m = a_c.join(m, on="a", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
    cd = pl.concat(cand_c).group_by("a").agg(pl.col("b").str.join(",").alias("candidate_entity_ids"))
    cd = a_c.join(cd, on="a", how="left").with_columns(pl.col("candidate_entity_ids").fill_null(""))
    n = pl.concat(sel_c).height
    print(f"{c}: S1={a_c.height:,}  with >=1 match={(m['matched_entity_ids'] != '').mean():.2%}  "
          f"avg matches/S1={n / a_c.height:.2f}", flush=True)
    match_parts.append(m); cand_parts.append(cd)
    del sel_c, cand_c; gc.collect()

match = pl.concat(match_parts).rename({"a": "source1_entity_id"})
cand = pl.concat(cand_parts).rename({"a": "source1_entity_id"})
assert match.height == s1.height == match["source1_entity_id"].n_unique(), "row count mismatch"
match.write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
cand.write_csv(f"{OUT}/candidate_pairs.tsv", separator="\t", quote_style="never")
print(f"\nwrote {match.height:,} rows to matching_results.tsv and candidate_pairs.tsv "
      f"| {time.time()-T0:.0f}s")
