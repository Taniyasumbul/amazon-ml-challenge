import json
import polars as pl

from decide import decide
from validation import HOLDOUT_FOLD, score_summary
from validation_data import load_validation_base

NORM = "/content/drive/MyDrive/amazon_norm"
DATA = "/content/student_resource/student_resource/dataset"

vp = pl.read_parquet(f"{NORM}/val_preds.parquet").filter(pl.col("split") == "holdout")
base, _ = load_validation_base(NORM, DATA, HOLDOUT_FOLD)
if vp.height == 0 or base.height == 0:
    raise ValueError("holdout split is empty; run train_model.py to create grouped predictions")

d_all = vp.select("a", "b", "p", "is_s3")
labels = vp.select("a", "b", "y")
with open(f"{NORM}/decision.json", encoding="utf-8") as f:
    decision = json.load(f)
selection = decide(d_all, decision)
selected = (selection.join(labels, on=["a", "b"], how="left")
                    .with_columns(pl.col("y").fill_null(0)))
scores = score_summary(base, selected)

print(f"untouched holdout: S1={base.height:,}, candidates={vp.height:,}")
print(f"official per-S1 macro F0.5={scores['overall']:.4f}")
print(f"country diagnostics={scores['country']}")
