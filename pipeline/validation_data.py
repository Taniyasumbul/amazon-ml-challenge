"""Load grouped S1 evaluation populations and their complete ground truth."""
import polars as pl

from validation import fold_expr


def load_validation_base(norm, data, fold):
    s1 = (pl.read_parquet(f"{norm}/train_s1.parquet", columns=["entity_id", "country"])
            .rename({"entity_id": "a"}))
    base = s1.filter(fold_expr() == fold).select("a", "country")

    gt = pl.read_csv(f"{data}/train/train_ground_truth.tsv", separator="\t",
                     infer_schema_length=0,
                     schema_overrides={"source1_entity_id": pl.Utf8,
                                       "matched_entity_ids": pl.Utf8})
    truth = (gt.rename({"source1_entity_id": "a", "matched_entity_ids": "b"})
               .join(base.select("a"), on="a", how="semi")
               .with_columns(pl.col("b").fill_null("").str.split(","))
               .explode("b")
               .with_columns(pl.col("b").str.strip_chars())
               .filter(pl.col("b") != "")
               .select("a", "b")
               .unique(subset=["a", "b"])
               .with_columns(pl.col("b").str.starts_with("S3-").cast(pl.Int8).alias("is_s3")))
    counts = truth.group_by("a").len().rename({"len": "n_true"})
    base = base.join(counts, on="a", how="left").with_columns(pl.col("n_true").fill_null(0))
    return base, truth
