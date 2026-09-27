# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** espDEVS
**Team Members:** Taniya Sumbul, Paras Sharma, Raghvendra Tyagi, Arnav Singh (KIET Group of Institutions, Ghaziabad)
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We built a four-stage pipeline: rule-based normalization (including Indic-script transliteration), multi-key IDF blocking re-ranked by a learned LambdaMART ranker, a LightGBM pairwise matcher over 57 engineered features, and a constrained decoder that assigns each Source 2/3 record to at most one Source 1 entity. The main technical contributions are (1) a learned candidate ranker that recovers most of the recall lost by a fixed-size candidate cut, especially for India, (2) **competition features** that compare each Source 1 entity's claim on a record with the best rival Source 1 entity, which targets the dataset's dominant error (same-name look-alike businesses), and (3) a country-agnostic feature set with no country input, so the model transfers to France, which has no training labels. Validation macro F₀.₅ is 0.956 (US 0.970, India 0.941); the leaderboard score of the final submission is **0.941**.

---

## 2. Methodology

### 2.1 Problem Analysis

All figures below were measured on the provided files.

| Finding | Evidence | Consequence |
|---|---|---|
| Each matched S2/S3 record belongs to exactly one S1 entity | 7,638,365 ground-truth links, 7,638,365 distinct matched IDs | Many-to-one assignment: a "one-parent" constraint at decode time |
| Country never differs within a matched group | 0 violations over all links | Blocking is partitioned by country |
| Singletons | 123,247 of 2,206,821 train S1 (5.58%) have no match | A correct empty prediction scores 1.0, so empty output must be a real option |
| Match counts | Mean 3.67 per matched S1; at most 5 from S2 and 6 from S3 | Per-source caps of 5 (S2) and 6 (S3) |
| Distractors | About 26% of S2/S3 records have no parent | The matcher must reject a large share of the pool |
| Scripts | S1 is always Latin; about 5% of India S2/S3 names are in one of 9 Indic scripts | Transliteration is needed before any string comparison |
| France | 259,452 test S1 (15%), absent from training | No country feature and no per-country constants |
| Scale | Test: 1.73M S1 × 4.89M S2 × 5.08M S3 | All-pairs comparison (~10¹³) is impossible, so blocking is mandatory |

**Noise observed in true matches.** Names: typos, legal-suffix churn (`Ltd`, `Inc`, `Pvt`), DBA wrappers (`X dba Y`), word reordering, accents, junk affixes (`--`, `>>`), domain collapse (`agroaditraj.com` ↔ `Aditraj Agro`), transliteration (`ग्रेट ट्रेडिंग` ↔ `Great Trading`), and **name replacement by a random token** (`Bullis Worldwide LLC` ↔ `Quonylaecto`, same address). Addresses: abbreviations (`Rd`/`Road`), state and city variants (`TX`/`Texas`, `Bangalore`/`Bengaluru`), component reordering, truncation to "number, city, state", digit corruption (`7315` → `315`), and empty addresses (~3.4%).

**Same-name look-alikes (found during error analysis).** Among validation candidate pairs whose normalized core names are *identical*, only 14.7% are true matches. Of the false ones, 80.3% belong to *another* S1 entity with the same name (e.g. several branches of one chain), and 17.7% have no owner at all. An S1 whose true match has an empty address has on average ~8 same-name candidates, of which ~2.8 are true. Separating same-name entities is therefore the core difficulty of this dataset.

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned candidate ranker + gradient-boosted pairwise classifier + constrained decoding.

**Core Innovation:** A LambdaMART ranker over per-family blocking evidence, so a small fixed K keeps most true matches; competition features computed against *all* Source 1 entities claiming a record, so the matcher can tell same-name businesses apart; and a fully country-agnostic feature set that lets one model serve US, India and France.

```
raw TSV → [1] normalize → [2] block (9 key families) → [3] rank (LambdaMART) → top-K
        → [4] 57 pair features (string, blocking, competition) → [5] LightGBM P(match)
        → [6] one-parent + threshold + caps → TSV
```

Every stage writes parquet, so each stage can be re-run independently.

---

## 3. Candidate Generation (Blocking)

### 3.1 Normalization (`src/normalize.py`, `src/translit.py`)

- **Indic → Latin:** all 9 Indic scripts in the data are code-point aligned, so each is folded onto Devanagari by offset, then passed through one Devanagari→Latin table with schwa deletion and nukta handling. Phonetically spelled acronyms are detected (`एलएलपी` → `llp`).
- **DBA resolution:** `X dba Y`, `d/b/a`, `doing business as`, `trading as`, `formerly`, `aka` and similar keep `Y` as the name; `X` is kept as `name_alias`.
- **Legal suffixes and honorifics** are removed into `name_core` (English, Indian and French forms, e.g. `llc`, `pvt`, `sarl`, `sas`).
- Letter-run merging (`l.l.c.` → `llc`), accent folding, `&` → `and`, address abbreviation expansion (`st` → `street`, `r` → `rue`, `opp` → `opposite`), and extraction of numeric tokens (house numbers, PIN/ZIP codes).

Outputs per record: `name_norm`, `name_core`, `name_alias`, `addr_norm`, `addr_toks`, `addr_nums`.

### 3.2 Blocking keys (`src/keys.py`, `src/blocking.py`)

A pair becomes a candidate if it shares **any** key from nine families:

| Family | Key | Handles |
|---|---|---|
| `n_full` | sorted core-name tokens | word order |
| `n_tok` | one rare core-name token | typos elsewhere in the name |
| `n_pair` | two rare core-name tokens | one typo |
| `n_skel` | consonant skeleton of the whole name | transliteration drift |
| `n_sktok` | consonant skeleton of one rare token | transliteration, single-token evidence |
| `n_anag` | sorted characters of the name | reordering and concatenation (domain names) |
| `na` | rare name token × address token | noise in either field |
| `a_pair` | two address tokens | completely different names |
| `a_num` | house number × address token | numeric anchor |

- **Rarity-driven token selection:** only a record's rarest tokens (document frequency over S1 ∪ target, per country) feed the name families. Rare tokens are both more discriminative and cheaper to join.
- **Address families mix rare and common tokens,** because S2/S3 addresses are often truncated to "number, city, state".
- The **consonant skeleton** folds sound pairs swapped by transliteration (`ph/f`, `bh/v/b/w`, `c/k/q`, `s/z/sh`), drops vowels and collapses repeats.
- Keys matching more than `MAX_KEY_DF = 120` target records are dropped as non-selective.
- For each (S1, target) pair, the IDF weights of shared keys are summed **per family**, giving nine evidence scores `f0…f8`, a family count `nfam`, and their sum `bscore`. The top 300 targets per S1 by `bscore` are kept for re-ranking.

### 3.3 Learned candidate ranker

Ranking by `bscore` alone dilutes recall: true matches found by fuzzy families fall below the cut. We trained a **LightGBM LambdaMART ranker** (`objective=lambdarank`, 200 trees, 63 leaves) on 23 features: the 9 family scores, `nfam`, `bscore`, and within-S1 relative features (score / best score for that S1, rank, candidate count, and each family score relative to its per-S1 maximum). Training data: uncapped candidates for 40,000 sampled S1 per country-source (US→S2, India→S2, India→S3), labelled from the ground truth, grouped by S1, with a fit/evaluation split by S1.

A logistic regression over the same evidence was tried first and was *worse* than `bscore`, because it scores pairs globally rather than within each S1's list.

**Recall on held-out S1 (share of true pairs kept in the top K):**

| Source pair | K | `bscore` | LambdaMART |
|---|---|---|---|
| US → S2 | 30 | 0.9776 | **0.9811** |
| US → S2 | 40 | 0.9797 | **0.9832** |
| India → S2 | 30 | 0.9150 | **0.9361** |
| India → S2 | 40 | 0.9309 | **0.9387** |
| India → S3 | 30 | 0.8922 | **0.9131** |
| India → S3 | 40 | 0.9005 | **0.9204** |

Uncapped ceilings (top 300): US→S2 0.9896, India→S2 0.9542, India→S3 0.9549.

### 3.4 Final candidate set

- **K = 30** per S1 per source for US and **K = 40** for India and France. France uses India's larger K because its recall cannot be measured.
- Target-side keys are built once per (country, source); S1 is processed in chunks of 25,000, with peak memory of about 7 GB.

| Country | S1 | S1→S2 pairs | S1→S3 pairs | Pairs per S1 per source |
|---|---|---|---|---|
| US | 663,106 | 19,419,184 | 19,153,614 | ~29 |
| India | 809,986 | 31,769,875 | 31,798,619 | ~39 |
| France | 259,452 | 10,230,744 | 10,234,664 | ~39 |
| **Total** | **1,732,544** | | | **122,606,700 pairs** |

99.99% of test S1 have at least one candidate. `candidate_pairs.tsv` is exactly this set: every pair the matching model scored.

**How true matches were protected:** nine complementary key families (name-only, address-only and mixed), the learned ranker, and measured recall on a 300,000-S1 training sample blocked against the *full* training S2/S3 pools:

| | S1→S2 recall | S1→S3 recall |
|---|---|---|
| US (K=30) | 0.9815 | 0.9784 |
| India (K=40) | 0.9380 | 0.9226 |

The ranker was never trained on US→S3, yet reaches 0.978 there, which is evidence that it transfers to unseen source pairs.

---

## 4. Matching Model

**Features used (57, all computed per candidate pair, none referring to country):**

- **Name features (16):** `rapidfuzz` ratio, token-sort ratio, token-set ratio, partial ratio and Jaro-Winkler on `name_core`; ratio on the no-space name (domain collapse); ratio on the full normalized name; ratio on the **consonant skeletons** (transliteration drift); alias match (DBA name of either side vs the other's core name); token Jaccard and containment; exact core-name match; exact full-name match; anagram-key match; length ratio; empty-name flag.
- **Name rarity and junk-name features (6):** how many records in each source share the exact core name (common-name detector); document frequency of each name's rarest token (a random replacement name such as `Quonylaecto` appears once); number of name tokens on each side.
- **Address features (13):** ratio and token-set ratio on `addr_norm`; address-token Jaccard; house/unit number Jaccard and shared-number count; primary house-number equality (compared after stripping leading zeros); near-miss house number (one is a suffix of the other, e.g. `315` vs `7315`); shared 5–6 digit PIN/ZIP; overlap of the last two address tokens (city/state); empty-address flags (either side, both sides, S1 side, target side).
- **Blocking and structural features (17):** the nine family IDF scores `f0…f8`, `nfam`, `bscore`, `bscore` relative to the S1's best, the ranker score and rank, the gap to the S1's best ranker score, the candidate count for the S1, and a source indicator (S2/S3).
- **Competition features (5), the key addition:** for record `b`, computed over **every** Source 1 entity whose candidate list contains `b` (training: blocking re-run for all 2.2M train S1; test: all 1.73M test S1): number of S1 entities claiming `b`; this S1's rank among them by ranker score; margin between this S1's ranker score and the best *other* claimant's; the same margin on raw blocking evidence; and the number of rival claimants with the *identical* core name. These let the model resolve same-name look-alikes by comparison instead of in isolation.

**Model type:** LightGBM binary classifier (MIT licence): learning rate 0.1, 127 leaves, `min_data_in_leaf` 200, feature and bagging fraction 0.8, L2 = 1.0, 500 rounds (best iteration 496), validation metric average precision, checkpoint every 100 rounds.

- **Training data:** candidates of 300,000 sampled train S1 (150,000 US, 150,000 India) blocked against the full train S2/S3 pools. That gives 20.5M labelled pairs with ~4.8% positives, split **by S1** (hash) into 16.4M training and 4.1M validation pairs.
- **Validation average precision:** 0.9967 (previous 43-feature model without competition features: 0.9925).
- **Most important features (gain):** ranker score, **competition margin (ranker)**, **competition rank**, **competition margin (blocking evidence)**, house-number Jaccard, full-name ratio, address token-set ratio, skeleton ratio, length ratio, address Jaccard. Three of the top four are competition features.

**Threshold selection method:** the decoder is chosen by maximizing macro F₀.₅, computed exactly as the official metric (per S1, singletons and S1 without candidates included), on the held-out validation S1:

1. **One-parent rule:** each S2/S3 record is kept only for the S1 that gives it the highest probability.
2. **Threshold:** keep pairs with P(match) ≥ t; t = 0.70 was best among 0.30–0.90.
3. **Caps:** at most 5 S2 and 6 S3 matches per S1.

Alternatives evaluated on validation and not adopted: separate thresholds per source (no gain), a per-S1 expected-F₀.₅ subset rule (worse), and a second-stage rescoring model using similarity to each S1's confident matches (+0.0013).

---

## 5. Results & Error Analysis

- **F₀.₅ score (macro, validation, t = 0.70), final model:** US 0.9699, India 0.9413 (mean **0.9556**)
- **Leaderboard, final model:** **0.941** (t = 0.70)

**Progression:**

| Model | Validation (US / India / mean) | Leaderboard |
|---|---|---|
| 43 features, t = 0.50 | | 0.922 |
| 43 features, t = 0.70 | 0.959 / 0.926 / 0.943 | 0.930 |
| 43 features, t = 0.80 | | 0.930 |
| **57 features incl. competition, t = 0.70** | **0.970 / 0.941 / 0.956** | **0.941** |

With the earlier model the score was flat between thresholds 0.70 and 0.80 and dropped when the threshold was loosened, consistent with the precision weighting of F₀.₅. The final gain came from the model, not the threshold: t = 0.70 remained optimal.

**Where the error came from before the competition features (validation, "what-if" analysis on the 43-feature model):**

| Scenario | US | India |
|---|---|---|
| 43-feature decoder | 0.959 | 0.926 |
| Remove every false positive | 0.970 | 0.944 |
| Add every rejected true match that was in the candidates | 0.983 | 0.955 |
| Perfect decisions on the candidate set (blocking ceiling) | 0.994 | 0.973 |

Model recall inside the candidate set was the largest loss, then precision, then (for India) blocking. Error analysis then showed that 80% of false exact-name candidates belong to another same-name S1 entity (section 2.1), which led to the competition features. The final model closes a large part of the model-error gap (US 0.959 → 0.970, India 0.926 → 0.941); the blocking ceiling (US 0.994, India 0.973) is unchanged.

- **Common false positives (wrong merges):** same-name records belonging to a *different* S1 entity (chain branches, common Indian business names), which are especially hard when the target address is empty or truncated. Pure decoys with plausible names account for the rest.
- **Common false negatives (missed matches):** 15,080 true pairs in the validation candidates were rejected; 88% received a middle probability (0.05–0.7). Typical cases: (a) exact or near-exact name with an **empty target address**, where name alone cannot rule out same-name look-alikes; (b) a **random replacement name** with a matching address (`Regional Church LLC` ↔ `Kororbilum`, same street); (c) **Indic-script names** with partial addresses (`First Care Limited` ↔ `ફર્સ્ટ કેર લિમિટેડ`); (d) heavily reformatted Indian addresses.

**Limitations and caveats.**

- The same S1 split served for early stopping (which never triggered) and for threshold selection, so the validation score is slightly optimistic. Country scores were averaged equally, while the official metric averages over all S1.
- The matcher is trained on 300,000 of 2.2M train S1. Rival information for the competition features comes from blocking *all* train S1, so validation reflects test-time competition, but the one-parent rule in validation still only sees sampled S1. The leaderboard score (0.941) is below validation (0.956).
- When blocking was re-run for all train S1, about 22% of the sampled pairs were not reproduced exactly, most likely because of ties in blocking scores at the top-K cut. The sampled pairs were therefore added to the rival table, so every training pair has competition features.
- France has no labels; its handling relies entirely on country-agnostic features and was not validated. Its match rate (96% of S1 with at least one match) is slightly above the training rate (~94%).
- The one-parent rule keeps exact probability ties; this affected 51 of 5.4M emitted matches.

---

## 6. Conclusion

A normalization → multi-key blocking → learned ranker → LightGBM → constrained-decoding pipeline reaches 0.956 validation and **0.941** leaderboard F₀.₅, using only the provided data and a CPU-only environment (Google Colab, 2 cores, 12.7 GB RAM). The largest single gain came from error analysis: after finding that most false candidates were same-name businesses owned by another Source 1 entity, we re-ran blocking for all training entities and added competition features, raising the leaderboard score from 0.930 to 0.941. The learned candidate ranker was the other key addition, raising India candidate recall by up to 2 points at fixed K. The remaining gap is dominated by India blocking recall (candidate ceiling 0.973), so further work would target new key families and larger K for India.

---

## Appendix

### A. Code Artefacts

```
code/business_entity_resolution/
├── src/
│   ├── translit.py        Indic → Latin transliteration
│   ├── normalize.py       field normalization
│   ├── keys.py            blocking key families, consonant skeleton
│   ├── blocking.py        document frequencies, key frames, candidate scoring
│   ├── prep.py
│   └── pipeline/
│       ├── make_ranker_data.py   labelled uncapped candidates for the ranker
│       ├── train_ranker.py       LambdaMART candidate ranker
│       ├── block_run.py          blocking + re-ranking + top-K (train sample and test)
│       ├── block_run_all.py      same, for all train S1 (rivals for competition features)
│       ├── comp_feats.py         competition features (rival S1 claimants per record)
│       ├── features.py           57 pair features
│       ├── train_feats.py        training features + labels
│       ├── train_model.py        LightGBM matcher
│       ├── val_preds.py          validation predictions
│       ├── decide.py / decide_eval.py / tune_decision.py   decoder and macro-F₀.₅ evaluation
│       ├── predict_test.py       test features + inference
│       └── make_submission.py    matching_results.tsv + candidate_pairs.tsv
├── README.md              exact run order and commands
└── requirements.txt       pinned versions (polars, lightgbm, rapidfuzz, numpy, scikit-learn, psutil)
```

Run order: normalize all six files → `make_ranker_data.py` → `train_ranker.py` → `block_run.py` (train sample, then test for every country and source) → `block_run_all.py` (all train S1) → `comp_feats.py` (train and test) → `train_feats.py` → `train_model.py` → `val_preds.py` → `decide_eval.py` → `predict_test.py` → `make_submission.py`. Every stage writes parquet and skips work that is already done, so interrupted runs resume.

**Compute (Colab CPU):** normalization ~23 min; ranker data ~40 min; test blocking ~2.5 h; blocking all train S1 ~2.4 h; competition features ~20 min; training features ~16 min; model training ~55 min; test inference ~3.5 h; submission writing ~4 min.

**Fair play:** no external data, APIs or services. Abbreviation, legal-suffix and transliteration tables are static linguistic resources. All document frequencies, weights, thresholds and models are derived from the provided training data. The only learned models are LightGBM (MIT licence); no LLMs or pretrained embeddings are used.

### B. Additional Results

**Test prediction sanity check:** mean predicted probability per group matches the training positive rate (US S2 0.057, US S3 0.062, India S2 0.041, India S3 0.043, France S2 0.047, France S3 0.053).

**Final submitted output (57-feature model, t = 0.70):**

| Country | S1 | S1 with ≥ 1 match | Mean matches per S1 |
|---|---|---|---|
| US | 663,106 | 94.5% | 3.31 |
| India | 809,986 | 93.5% | 3.12 |
| France | 259,452 | 96.2% | 3.50 |

The previous submission (43 features) had 3.17 / 3.03 / 3.31 matches per S1; the final model finds more true matches at the same threshold, closer to the training mean of 3.46.
