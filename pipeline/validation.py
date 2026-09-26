"""Shared grouped validation splits and official per-S1 scoring."""
import polars as pl

FOLD_SEED = 42
EARLY_STOP_FOLD = 0
TUNE_FOLD = 1
HOLDOUT_FOLD = 2
N_FOLDS = 10


def fold_expr(column="a"):
    return pl.col(column).hash(seed=FOLD_SEED) % N_FOLDS


def score_per_s1(base, selected):
    """Score selected candidate pairs against a complete S1 evaluation set."""
    required = {"a", "country", "n_true"}
    if not required.issubset(base.columns):
        raise ValueError(f"base is missing columns: {sorted(required - set(base.columns))}")
    if not {"a", "y"}.issubset(selected.columns):
        raise ValueError("selected must contain S1 IDs and binary truth labels in columns 'a' and 'y'")

    counts = selected.group_by("a").agg(
        pl.len().alias("n_pred"), pl.col("y").sum().alias("tp"))
    scored = (base.join(counts, on="a", how="left")
                  .with_columns(pl.col("n_pred").fill_null(0), pl.col("tp").fill_null(0)))
    denominator = (1.25 * pl.col("tp")
                   + 0.25 * (pl.col("n_true") - pl.col("tp"))
                   + (pl.col("n_pred") - pl.col("tp")))
    f05 = (pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
             .when((pl.col("n_true") == 0) | (pl.col("n_pred") == 0) | (pl.col("tp") == 0)).then(0.0)
             .otherwise(1.25 * pl.col("tp") / denominator))
    return scored.with_columns(f05.alias("f05"))


def score_summary(base, selected):
    scored = score_per_s1(base, selected)
    by_country = scored.group_by("country").agg(pl.col("f05").mean().alias("f05"))
    return {
        "overall": scored["f05"].mean(),
        "country": {country: score for country, score in by_country.iter_rows()},
    }
