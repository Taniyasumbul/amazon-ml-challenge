import os, glob, json, time, gc
import polars as pl

from decide import decide

NORM = "/content/drive/MyDrive/amazon_norm"
OUT = f"{NORM}/output"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
with open(f"{NORM}/decision.json", encoding="utf-8") as f:
    decision = json.load(f)
print(f"decision: {decision}", flush=True)

s1 = (pl.read_parquet(f"{NORM}/test_s1.parquet", columns=["entity_id", "country"])
        .rename({"entity_id": "a"}))
if s1["country"].null_count():
    raise ValueError("test S1 contains rows without a country; cannot route predictions safely")
countries = sorted(s1["country"].unique().to_list())
match_parts, candidate_parts = [], []

for country in countries:
    parts = []
    for source in ("2", "3"):
        files = sorted(glob.glob(f"{NORM}/preds/test_{country}_S{source}_*.parquet"))
        if not files:
            raise FileNotFoundError(f"no prediction files found for {country} S{source}")
        part = pl.read_parquet(files)
        parts.append(part)
        print(f"  {country} S{source}: {part.height:,} candidate pairs | "
              f"{time.time()-T0:.0f}s", flush=True)

    candidates = pl.concat(parts, how="vertical")
    selected = decide(candidates, decision)
    a_country = s1.filter(pl.col("country") == country).select("a")

    matched = (selected.group_by("a")
                       .agg(pl.col("b").unique().sort().alias("_ids"))
                       .with_columns(pl.col("_ids").list.join(",").alias("matched_entity_ids"))
                       .select("a", "matched_entity_ids"))
    matched = (a_country.join(matched, on="a", how="left")
                        .with_columns(pl.col("matched_entity_ids").fill_null("")))
    candidate_rows = (candidates.select("a", "b").group_by("a")
                                .agg(pl.col("b").unique().sort().alias("_ids"))
                                .with_columns(pl.col("_ids").list.join(",").alias("candidate_entity_ids"))
                                .select("a", "candidate_entity_ids"))
    candidate_rows = (a_country.join(candidate_rows, on="a", how="left")
                               .with_columns(pl.col("candidate_entity_ids").fill_null("")))
    if matched.height != a_country.height or candidate_rows.height != a_country.height:
        raise ValueError(f"output row count mismatch for {country}")

    print(f"{country}: S1={a_country.height:,}, with matches="
          f"{(matched['matched_entity_ids'] != '').mean():.2%}", flush=True)
    match_parts.append(matched)
    candidate_parts.append(candidate_rows)
    del parts, candidates, selected, matched, candidate_rows; gc.collect()

match = pl.concat(match_parts).rename({"a": "source1_entity_id"}).sort("source1_entity_id")
candidate_pairs = (pl.concat(candidate_parts)
                     .rename({"a": "source1_entity_id"})
                     .sort("source1_entity_id"))
if match.height != s1.height or match["source1_entity_id"].n_unique() != s1.height:
    raise ValueError("matching_results.tsv must have exactly one row for every test S1")
if (candidate_pairs.height != s1.height
        or candidate_pairs["source1_entity_id"].n_unique() != s1.height):
    raise ValueError("candidate_pairs.tsv must have exactly one row for every test S1")

match.write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
candidate_pairs.write_csv(f"{OUT}/candidate_pairs.tsv", separator="\t", quote_style="never")
print(f"\nwrote {match.height:,} rows to matching_results.tsv and candidate_pairs.tsv "
      f"| {time.time()-T0:.0f}s")
