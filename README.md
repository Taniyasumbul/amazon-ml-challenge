# Business Entity Resolution: Amazon ML Challenge 2026 (team espDEVS)

Regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the
challenge dataset. Methodology: see `Documentation_template.md` at the root of the package.

## Layout

```
src/                     translit.py, normalize.py, keys.py, blocking.py, prep.py
src/pipeline/            all pipeline stages (run in the order below)
models/                  trained artefacts: block_ranker_lgb.txt, match_model_lgb.txt, decision.json
requirements.txt         pinned versions
```

## Environment

Python 3.12+, CPU only (developed on Google Colab: 2 cores, 12.7 GB RAM).

```bash
pip install -r requirements.txt
```

Paths are read from environment variables (defaults are the Colab paths used during the challenge):

| Variable | Meaning | Default |
|---|---|---|
| `ER_DATA` | challenge `dataset/` folder (train/, test/) | `/content/student_resource/student_resource/dataset` |
| `ER_WORK` | working folder for all intermediate files and outputs | `/content/drive/MyDrive/amazon_norm` |
| `ER_SRC` | this package's `src/` folder | `/content/business_entity_resolution/src` |

Some stage scripts (`block_run.py`, `train_feats.py`, `predict_test.py` and the feature module)
still contain these defaults as constants at the top of the file; edit them there if your
folders differ.

`PYTHONHASHSEED=0` must be set for every blocking stage, so key hashes are reproducible.

## Run order

| # | Step | Command | Output (in `$ER_WORK`) | Time (Colab CPU) |
|---|---|---|---|---|
| 1 | Normalize all six source files | `normalize.normalize_frame` on each TSV, saved as `{train,test}_s{1,2,3}.parquet` | normalized parquet | ~23 min |
| 2 | Ranker training data | `PYTHONHASHSEED=0 python src/pipeline/make_ranker_data.py US 2` (also `India 2`, `India 3`) | `ranker_data/` | ~40 min |
| 3 | Train candidate ranker | `python src/pipeline/train_ranker.py` | `block_ranker_lgb.txt` | ~10 min |
| 4 | Training sample | 150,000 US + 150,000 India train S1 (seed 11) saved as `train_block_ids.parquet` | ids | seconds |
| 5 | Blocking, train | `PYTHONHASHSEED=0 python src/pipeline/block_run.py train <country> <src> <K> $ER_WORK/train_block_ids.parquet` for US (K=30) and India (K=40), sources 2 and 3 | `blocks/train_*` | ~35 min |
| 6 | Blocking, test | `PYTHONHASHSEED=0 python src/pipeline/block_run.py test <country> <src> <K>` for every test country (US K=30, others K=40), sources 2 and 3 | `blocks/test_*` = candidate set | ~2.5 h |
| 7 | Training features | `python src/pipeline/train_feats.py <country> <src>` for US/India × 2/3 | `feats/` | ~12 min |
| 8 | Train matcher | `python src/pipeline/train_model.py` | `match_model_lgb.txt` | ~77 min |
| 9 | Validation predictions | `python src/pipeline/val_preds.py` | `val_preds.parquet` | ~9 min |
| 10 | Decision rule + validation F0.5 | `python src/pipeline/decide_eval.py` | `decision.json` | ~2 min |
| 11 | Test predictions | `python src/pipeline/predict_test.py <country> <src>` for every test country and source | `preds/` | ~5 h |
| 12 | Submission files | `python src/pipeline/make_submission.py` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` | ~5 min |

Every stage skips work whose output already exists, so an interrupted run can simply be restarted.

To reuse the shipped models instead of retraining, copy `models/*` into `$ER_WORK` and run
steps 1, 6, 11 and 12 only.

## Other scripts

- `decide.py`, `tune_decision.py`: alternative decoders (per-source thresholds, expected-F0.5 rule) evaluated on validation; not used for the submission.
- `stage2_exp.py`: second-stage rescoring experiment; not used for the submission.
- `make_matches.py`: rewrites `matching_results.tsv` only, for trying other thresholds.

## Data use

Only the provided challenge files are used. No external data, APIs or services.
Models: LightGBM (MIT licence). No LLMs or pretrained embeddings.
