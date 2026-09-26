# Business Entity Resolution

- src/: normalization, blocking keys
- pipeline/: all pipeline scripts
- models/: ranker, match model, rule

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
fold 2. Both use the official per-S1 F0.5 mean over the full evaluation
population, including singleton S1s and S1s with no candidates. Country scores
are diagnostic only. `pipeline/val_preds.py` summarizes the saved predictions;
it does not regenerate them.

The tuner writes one decision config consumed by `pipeline/decide.py` for both
validation and `pipeline/make_submission.py` / `pipeline/make_matches.py` for
inference. The checked-in `models/decision.json` is a fallback example; use the
config produced by the current tuning run for submission generation.

The pipeline scripts currently expect the prepared feature/prediction
artifacts and Colab paths shown in each script. The complete blocker-to-feature
artifact runner remains to be documented.
