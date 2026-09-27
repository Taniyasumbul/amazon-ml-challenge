"""Write output/matching_results.tsv (and candidate_pairs.tsv) from test predictions.

Usage:  python make_submission.py                 # both files
        python make_submission.py --matches-only  # matching_results.tsv only (low memory)

Decoder (same rule as decide_eval.py): for each country and source,
  1. one-parent: each S2/S3 record is kept only for its highest-probability S1
     (exact ties broken deterministically, so a record is never given to two S1),
  2. keep pairs with p >= t,
  3. at most caps[S2] / caps[S3] matches per S1.
Countries are read from the test data, not hard-coded. Every submission invariant is
checked before anything is written.
"""
import os, sys, glob, json, time, gc
import polars as pl

NORM = os.environ.get("ER_WORK", "/content/drive/MyDrive/amazon_norm")
DATA = os.environ.get("ER_DATA", "/content/student_resource/student_resource/dataset")
OUT = f"{NORM}/output"
MATCHES_ONLY = "--matches-only" in sys.argv
os.makedirs(OUT, exist_ok=True)
T0 = time.time()


def log(msg):
    print(f"{msg} | {time.time() - T0:.0f}s", flush=True)


def load_decision():
    dec = json.load(open(f"{NORM}/decision.json"))
    missing = {"t", "one_parent", "caps"} - dec.keys()
    if missing or not {"S2", "S3"} <= dec["caps"].keys():
        sys.exit(f"decision.json has the wrong format (missing {missing}): {dec}")
    return dec


def decode(d, dec, cap):
    if dec["one_parent"]:
        d = (d.sort(["b", "p", "a"], descending=[False, True, False])
              .with_columns(pl.int_range(pl.len()).over("b").alias("rb"))
              .filter(pl.col("rb") == 0))
    d = d.filter(pl.col("p") >= dec["t"])
    d = (d.with_columns(pl.col("p").rank("ordinal", descending=True).over("a").alias("rk"))
          .filter(pl.col("rk") <= cap))
    return d.select("a", "b")


dec = load_decision()
log(f"decision: {dec}")
s1 = pl.read_parquet(f"{NORM}/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "a"})
countries = sorted(s1["country"].unique().to_list())
log(f"countries in test data: {countries}")

sel_parts, cand_parts = [], []
for c in countries:
    for s in ["2", "3"]:
        files = sorted(glob.glob(f"{NORM}/preds/test_{c}_S{s}_*.parquet"))
        if not files:
            sys.exit(f"no predictions for {c} S{s}; run predict_test.py {c} {s} first")
        d = pl.read_parquet(files, columns=["a", "b", "p"])
        sel = decode(d, dec, dec["caps"][f"S{s}"])
        sel_parts.append(sel)
        if not MATCHES_ONLY:
            cand_parts.append(d.select("a", "b"))
        log(f"  {c} S{s}: {d.height:,} candidates -> {sel.height:,} matches")
        del d; gc.collect()

sel = pl.concat(sel_parts)
del sel_parts; gc.collect()

# ---- invariants, checked before writing ----
ids23 = pl.concat([pl.read_csv(f"{DATA}/test/test_source{s}.tsv", separator="\t",
                               columns=["entity_id"], infer_schema_length=0) for s in (2, 3)])
ids23 = ids23.rename({"entity_id": "b"})
problems = []
if sel.join(ids23, on="b", how="anti").height:
    problems.append("matched IDs that are not test S2/S3 records")
if sel["b"].n_unique() != sel.height:
    problems.append("an S2/S3 record is matched to more than one S1")
if sel.join(s1.select("a"), on="a", how="anti").height:
    problems.append("matches for S1 IDs that are not in the test set")
if problems:
    sys.exit("REFUSING TO WRITE: " + "; ".join(problems))


def to_rows(pairs, col):
    agg = pairs.group_by("a").agg(pl.col("b").str.join(",").alias(col))
    return (s1.select("a").join(agg, on="a", how="left")
              .with_columns(pl.col(col).fill_null(""))
              .rename({"a": "source1_entity_id"}))


match = to_rows(sel, "matched_entity_ids")
assert match.height == s1.height == match["source1_entity_id"].n_unique(), "row count mismatch"
for c in countries:
    a_c = s1.filter(pl.col("country") == c).select("a")
    n = sel.join(a_c, on="a", how="semi").height
    m_c = match.filter(pl.col("source1_entity_id").is_in(a_c["a"].implode()))
    log(f"{c}: S1={a_c.height:,}  with >=1 match={(m_c['matched_entity_ids'] != '').mean():.2%}  "
        f"avg matches/S1={n / a_c.height:.2f}")
match.write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
log(f"wrote matching_results.tsv ({match.height:,} rows)")

if not MATCHES_ONLY:
    cand_pairs = pl.concat(cand_parts)
    del cand_parts; gc.collect()
    # matches are a subset of candidates by construction (both come from the same prediction files)
    cand = to_rows(cand_pairs, "candidate_entity_ids")
    cand.write_csv(f"{OUT}/candidate_pairs.tsv", separator="\t", quote_style="never")
    log(f"wrote candidate_pairs.tsv ({cand.height:,} rows, {cand_pairs.height:,} pairs)")
