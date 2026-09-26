import json, time
import polars as pl

from decide import decide
from validation import TUNE_FOLD, score_summary
from validation_data import load_validation_base

NORM = "/content/drive/MyDrive/amazon_norm"
DATA = "/content/student_resource/student_resource/dataset"
T0 = time.time()

vp = pl.read_parquet(f"{NORM}/val_preds.parquet").filter(pl.col("split") == "tune")
base, truth = load_validation_base(NORM, DATA, TUNE_FOLD)
if vp.height == 0 or base.height == 0:
    raise ValueError("tuning split is empty; run train_model.py to create grouped predictions")
duplicates = vp.group_by("a", "b").len().filter(pl.col("len") > 1)
if duplicates.height:
    raise ValueError(f"tuning candidates contain {duplicates.height:,} duplicate S1/target pairs")

d_all = vp.select("a", "b", "p", "is_s3")
labels = vp.select("a", "b", "y")


def evaluate(cfg):
    selected = (decide(d_all, cfg)
                .join(labels, on=["a", "b"], how="left")
                .with_columns(pl.col("y").fill_null(0)))
    return score_summary(base, selected)


def tune(one_parent, caps):
    results = []
    for t2 in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for t3 in [0.4, 0.5, 0.6, 0.7, 0.8]:
            cfg = {"mode": "threshold", "thresholds": {"S2": t2, "S3": t3},
                   "one_parent": one_parent, "caps": caps, "min_score": 0.01}
            results.append((cfg, evaluate(cfg)))
    return results


fixed_caps = {"S2": 5, "S3": 6}
results = tune(one_parent=True, caps=fixed_caps)
best = max(results, key=lambda result: result[1]["overall"])
baseline = next(result for result in results
                if result[0]["thresholds"] == {"S2": 0.5, "S3": 0.5})
best_common = max((result for result in results
                   if result[0]["thresholds"]["S2"] == result[0]["thresholds"]["S3"]),
                  key=lambda result: result[1]["overall"])

print(f"tuned {len(results)} threshold pairs on {base.height:,} S1s | {time.time()-T0:.0f}s")
print(f"selected: overall={best[1]['overall']:.4f} country={best[1]['country']} config={best[0]}")
print(f"baseline: overall={baseline[1]['overall']:.4f} country={baseline[1]['country']}")
print(f"best shared threshold: {best_common[0]['thresholds']} "
      f"overall={best_common[1]['overall']:.4f}")

decoder_ablation = []
for name, overrides in [
    ("without one-parent", {"one_parent": False}),
    ("without caps", {"caps": {}}),
    ("without parent or caps", {"one_parent": False, "caps": {}}),
]:
    cfg = {**best[0], **overrides}
    scores = evaluate(cfg)
    decoder_ablation.append({"name": name, "config": cfg, "scores": scores})
    print(f"{name}: overall={scores['overall']:.4f} country={scores['country']} "
          "(selected thresholds held fixed)")

candidate_oracle = score_summary(base, vp.filter(pl.col("y") == 1).select("a", "b", "y"))
print(f"candidate-set oracle: overall={candidate_oracle['overall']:.4f} "
      f"country={candidate_oracle['country']}")

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

source_oracles = []
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
        source_oracles.append({"country": country, "source_pair": source, "f05": score})
        print(f"candidate-set oracle {country} {source}: {score:.4f}")

with open(f"{NORM}/decision.json", "w", encoding="utf-8") as f:
    json.dump(best[0], f, indent=2)
with open(f"{NORM}/decision_ablation.json", "w", encoding="utf-8") as f:
    json.dump({"best_common_threshold": best_common[0],
               "decoder_ablation": decoder_ablation,
               "candidate_oracle": candidate_oracle,
               "candidate_recall": recall.to_dicts(),
               "source_oracles": source_oracles}, f, indent=2)
print(f"saved decision.json: {best[0]} "
      f"(gain vs baseline: {best[1]['overall']-baseline[1]['overall']:+.4f})")
