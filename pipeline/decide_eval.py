import json
import polars as pl

from decide import decide
from validation import HOLDOUT_FOLD, score_summary
from validation_data import load_validation_base

NORM = "/content/drive/MyDrive/amazon_norm"
DATA = "/content/student_resource/student_resource/dataset"

vp = pl.read_parquet(f"{NORM}/val_preds.parquet").filter(pl.col("split") == "holdout")
base, truth = load_validation_base(NORM, DATA, HOLDOUT_FOLD)
if vp.height == 0 or base.height == 0:
    raise ValueError("holdout split is empty; run train_model.py to create grouped predictions")

duplicates = vp.group_by("a", "b").len().filter(pl.col("len") > 1)
if duplicates.height:
    raise ValueError(f"validation candidates contain {duplicates.height:,} duplicate S1/target pairs")

d_all = vp.select("a", "b", "p", "is_s3")
labels = vp.select("a", "b", "y")


def evaluate(selection):
    selected = (selection.join(labels, on=["a", "b"], how="left")
                         .with_columns(pl.col("y").fill_null(0)))
    return score_summary(base, selected)


oracle = evaluate(vp.filter(pl.col("y") == 1).select("a", "b"))
with open(f"{NORM}/decision.json", encoding="utf-8") as f:
    decision = json.load(f)
chosen = evaluate(decide(d_all, decision))

print(f"holdout S1={base.height:,} | candidates={vp.height:,}")
print(f"candidate-set oracle: overall={oracle['overall']:.4f} "
      f"country={oracle['country']}")
print(f"selected decision:    overall={chosen['overall']:.4f} "
      f"country={chosen['country']}  config={decision}")

candidate_hits = (vp.group_by("country", "is_s3")
                    .agg(pl.col("y").sum().alias("candidate_true")))
truth_counts = (truth.join(base.select("a", "country"), on="a", how="inner")
                     .group_by("country", "is_s3").len()
                     .rename({"len": "true_pairs"}))
recall = (truth_counts.join(candidate_hits, on=["country", "is_s3"], how="left")
                      .with_columns(pl.col("candidate_true").fill_null(0),
                                    (pl.col("candidate_true") / pl.col("true_pairs")).alias("candidate_recall"))
                      .sort(["country", "is_s3"]))
print("candidate recall by country and source pair:")
for row in recall.iter_rows(named=True):
    source = "S1→S3" if row["is_s3"] else "S1→S2"
    print(f"  {row['country']:8s} {source}: {row['candidate_true']:,}/"
          f"{row['true_pairs']:,} = {row['candidate_recall']:.4f}")

print("candidate-set oracle by country and source pair:")
for country in sorted(base["country"].unique().to_list()):
    country_base = base.filter(pl.col("country") == country).select("a", "country")
    country_truth = truth.join(country_base.select("a"), on="a", how="semi")
    for source_flag in (0, 1):
        source_truth = (country_truth.filter(pl.col("is_s3") == source_flag)
                                   .group_by("a").len().rename({"len": "n_true"}))
        source_base = (country_base.join(source_truth, on="a", how="left")
                                  .with_columns(pl.col("n_true").fill_null(0)))
        source_oracle = vp.filter((pl.col("country") == country)
                                  & (pl.col("is_s3") == source_flag)
                                  & (pl.col("y") == 1)).select("a", "b", "y")
        score = score_summary(source_base, source_oracle)["overall"]
        source = "S1→S3" if source_flag else "S1→S2"
        print(f"  {country:8s} {source}: {score:.4f}")

ablations = [
    ("without one-parent", {**decision, "one_parent": False}),
    ("without caps", {**decision, "caps": {}}),
    ("without parent or caps", {**decision, "one_parent": False, "caps": {}}),
]
ablation_path = f"{NORM}/decision_ablation.json"
try:
    with open(ablation_path, encoding="utf-8") as f:
        shared_threshold = json.load(f)["best_common_threshold"]
    ablations.insert(0, ("shared S2/S3 threshold", shared_threshold))
except FileNotFoundError:
    print("no shared-threshold tuning artifact; run tune_decision.py for that ablation")
print("decoder ablations on the same holdout:")
for name, cfg in ablations:
    score = evaluate(decide(d_all, cfg))
    print(f"  {name}: overall={score['overall']:.4f} country={score['country']}")
