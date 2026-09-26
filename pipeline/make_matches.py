"""Write only matching_results.tsv using the shared submission decoder."""
import os, glob, json, gc
import polars as pl

from decide import decide

NORM = "/content/drive/MyDrive/amazon_norm"
OUT = f"{NORM}/output"
os.makedirs(OUT, exist_ok=True)
with open(f"{NORM}/decision.json", encoding="utf-8") as f:
    decision = json.load(f)

s1 = (pl.read_parquet(f"{NORM}/test_s1.parquet", columns=["entity_id", "country"])
        .rename({"entity_id": "a"}))
if s1["country"].null_count():
    raise ValueError("test S1 contains rows without a country; cannot route predictions safely")
parts = []
for country in sorted(s1["country"].unique().to_list()):
    predictions = []
    for source in ("2", "3"):
        files = sorted(glob.glob(f"{NORM}/preds/test_{country}_S{source}_*.parquet"))
        if not files:
            raise FileNotFoundError(f"no prediction files found for {country} S{source}")
        predictions.append(pl.read_parquet(files))
    selected = decide(pl.concat(predictions, how="vertical"), decision)
    anchors = s1.filter(pl.col("country") == country).select("a")
    matched = (selected.group_by("a")
                       .agg(pl.col("b").unique().sort().alias("_ids"))
                       .with_columns(pl.col("_ids").list.join(",").alias("matched_entity_ids"))
                       .select("a", "matched_entity_ids"))
    parts.append(anchors.join(matched, on="a", how="left")
                        .with_columns(pl.col("matched_entity_ids").fill_null("")))
    del predictions, selected, matched; gc.collect()

result = pl.concat(parts).rename({"a": "source1_entity_id"}).sort("source1_entity_id")
if result.height != s1.height or result["source1_entity_id"].n_unique() != s1.height:
    raise ValueError("matching_results.tsv must have exactly one row for every test S1")
result.write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
print(f"wrote {result.height:,} rows to matching_results.tsv")
