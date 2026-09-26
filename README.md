# Business Entity Resolution

- src/: normalization, blocking keys
- pipeline/: all pipeline scripts
- models/: ranker, match model, rule

## Google Colab run

The scripts use `/content/drive/MyDrive/amazon_norm` for generated artifacts
and `/content/student_resource/student_resource/dataset` for competition data.
Mount Drive and extract `student_resource.zip` so that directory contains
`train/` and `test/`. Then clone the PR branch and install dependencies:

```bash
git clone https://github.com/Taniyasumbul/amazon-ml-challenge.git /content/amazon-ml-challenge
cd /content/amazon-ml-challenge
git fetch origin pull/2/head:pr-2
git switch pr-2
pip install -r requirements.txt
```

Normalize the raw TSVs:

```bash
python /content/amazon-ml-challenge/src/prep.py \
  /content/student_resource/student_resource/dataset \
  /content/drive/MyDrive/amazon_norm
```

Create a reproducible training sample. Reuse this file for every training
candidate-generation command; validation scores the same S1 population. The
sample size follows the current Colab workflow and can be reduced for a smaller
runtime.

```python
import polars as pl

norm = "/content/drive/MyDrive/amazon_norm"
s1 = pl.read_parquet(f"{norm}/train_s1.parquet", columns=["entity_id", "country"])
parts = []
for country in ["US", "India"]:
    rows = s1.filter(pl.col("country") == country)
    if rows.height:
        parts.append(rows.sample(n=min(150_000, rows.height), seed=11)
                     .select("entity_id"))
pl.concat(parts).unique().write_parquet(f"{norm}/train_block_ids.parquet")
```

Build ranked candidates, features, and the model. The current training data uses
`US` and `India`; add any other training countries present in your data. If you
change the sample or top-K, regenerate the matching files under both `blocks/`
and `feats/` so cached artifacts stay consistent.

```bash
cd /content/amazon-ml-challenge
for country in US India; do
  k=40
  if [ "$country" = US ]; then k=30; fi
  for source in 2 3; do
    PYTHONHASHSEED=0 python -u pipeline/build_candidates.py train "$country" "$source" "$k" \
      --ids-file /content/drive/MyDrive/amazon_norm/train_block_ids.parquet
    python -u pipeline/train_feats.py "$country" "$source"
  done
done
python -u pipeline/train_model.py
python -u pipeline/tune_decision.py
python -u pipeline/decide_eval.py
```

The holdout score estimates performance on the sampled training S1s. It helps
compare choices made on the tuning split; it does not guarantee a leaderboard
gain.

After validation, generate test candidates and predictions for each test
country, then write both submission files:

```bash
for country in US India France; do
  k=40
  if [ "$country" = US ]; then k=30; fi
  for source in 2 3; do
    PYTHONHASHSEED=0 python -u pipeline/build_candidates.py test "$country" "$source" "$k"
    python -u pipeline/predict_test.py "$country" "$source"
  done
done
python -u pipeline/make_matches.py
```

The outputs are `matching_results.tsv` and `candidate_pairs.tsv` under
`/content/drive/MyDrive/amazon_norm/output/`.

## Validation and decoding

`pipeline/train_model.py` assigns each S1 ID to one deterministic hash group:

- fold 0: LightGBM early stopping
- fold 1: decision-threshold tuning
- fold 2: final local report holdout
- folds 3–9: development-model training

It writes `val_preds.parquet` from the development model before refitting the
deployment model on all labeled candidate pairs at the selected iteration
count. Do not use the refit model to score the saved holdout predictions.

Run `pipeline/tune_decision.py` on fold 1, then `pipeline/decide_eval.py` on
fold 2. Both use the official per-S1 F0.5 mean over the evaluation population,
including singleton S1s and S1s with no candidates. If
`train_block_ids.parquet` exists, that population is restricted to those S1s so
it matches a sampled candidate-generation run; otherwise all training S1s are
included. Country scores are diagnostic only. Candidate recall/oracle and
decoder ablations are reported on the tuning fold; `decide_eval.py` reports
only the selected config on the untouched holdout. `pipeline/val_preds.py`
summarizes the saved predictions; it does not regenerate them.

The tuner writes one decision config consumed by `pipeline/decide.py` for both
validation and `pipeline/make_submission.py` / `pipeline/make_matches.py` for
inference. The checked-in `models/decision.json` is a fallback example; use the
config produced by the current tuning run for submission generation.

The candidate builder uses the checked-in blocking ranker at
`models/block_ranker_lgb.txt`; fixed Colab paths can be changed in the scripts
when running outside Colab.
