import polars as pl

NORM = "/content/drive/MyDrive/amazon_norm"
path = f"{NORM}/val_preds.parquet"
preds = pl.read_parquet(path)
required = {"a", "b", "y", "p", "is_s3", "country", "split"}
missing = required - set(preds.columns)
if missing:
    raise ValueError(f"{path} is missing grouped validation columns: {sorted(missing)}; rerun train_model.py")

for row in (preds.group_by("split", "country")
                  .agg(pl.len().alias("pairs"), pl.col("y").sum().alias("positive_pairs"))
                  .sort(["split", "country"]).iter_rows(named=True)):
    rate = row["positive_pairs"] / max(row["pairs"], 1)
    print(f"{row['split']:8s} {row['country']:8s}: {row['pairs']:,} pairs, "
          f"{row['positive_pairs']:,} positive ({rate:.2%})")
