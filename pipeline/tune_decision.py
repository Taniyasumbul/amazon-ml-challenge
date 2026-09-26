import json, time
import polars as pl

from decide import decide
from validation import TUNE_FOLD, score_summary
from validation_data import load_validation_base

NORM = "/content/drive/MyDrive/amazon_norm"
DATA = "/content/student_resource/student_resource/dataset"
T0 = time.time()

vp = pl.read_parquet(f"{NORM}/val_preds.parquet").filter(pl.col("split") == "tune")
base, _ = load_validation_base(NORM, DATA, TUNE_FOLD)
if vp.height == 0 or base.height == 0:
    raise ValueError("tuning split is empty; run train_model.py to create grouped predictions")

d_all = vp.select("a", "b", "p", "is_s3")
labels = vp.select("a", "b", "y")


def evaluate(cfg):
    selected = (decide(d_all, cfg)
                .join(labels, on=["a", "b"], how="left")
                .with_columns(pl.col("y").fill_null(0)))
    return score_summary(base, selected)


fixed = {"mode": "threshold", "one_parent": True,
         "caps": {"S2": 5, "S3": 6}, "min_score": 0.01}
results = []
for t2 in [0.6, 0.65, 0.7, 0.75, 0.8]:
    for t3 in [0.6, 0.65, 0.7, 0.75, 0.8]:
        cfg = {**fixed, "thresholds": {"S2": t2, "S3": t3}}
        results.append((cfg, evaluate(cfg)))

baseline = next(r for r in results
                if r[0]["thresholds"] == {"S2": 0.7, "S3": 0.7})
best_common = max((result for result in results
                   if result[0]["thresholds"]["S2"] == result[0]["thresholds"]["S3"]),
                  key=lambda result: result[1]["overall"])
results.sort(key=lambda result: -result[1]["overall"])
print(f"evaluated {len(results)} rules on {base.height:,} tune S1s | {time.time()-T0:.0f}s")
print(f"baseline: overall={baseline[1]['overall']:.4f} country={baseline[1]['country']}")
print("top 8:")
for cfg, scores in results[:8]:
    print(f"  overall={scores['overall']:.4f} country={scores['country']}  {cfg}")

best = results[0][0]
path = f"{NORM}/decision.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(best, f, indent=2)
with open(f"{NORM}/decision_ablation.json", "w", encoding="utf-8") as f:
    json.dump({"best_common_threshold": best_common[0]}, f, indent=2)
print(f"saved decision.json: {best} "
      f"(gain vs baseline: {results[0][1]['overall']-baseline[1]['overall']:+.4f})")
print(f"best shared threshold on tuning split: {best_common[0]['thresholds']}")
