"""Stage 1: read raw TSVs, normalise, cache as parquet."""
from __future__ import annotations
import sys, time
from pathlib import Path
import polars as pl
from normalize import normalize_frame

SCHEMA = {"entity_id": pl.Utf8, "business_name": pl.Utf8,
          "business_address": pl.Utf8, "country": pl.Utf8}


def read_raw(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", schema_overrides=SCHEMA,
                       quote_char=None, has_header=True)


def main(data_dir: str, work_dir: str) -> None:
    data, work = Path(data_dir), Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for src in (1, 2, 3):
            src_path = data / split / f"{split}_source{src}.tsv"
            dst = work / f"{split}_s{src}.parquet"
            if dst.exists():
                print(f"skip {dst.name}"); continue
            t = time.time()
            df = normalize_frame(read_raw(src_path))
            df.write_parquet(dst, compression="zstd")
            print(f"{dst.name}: {df.height:,} rows in {time.time()-t:.1f}s")
    gt_src = data / "train" / "train_ground_truth.tsv"
    gt_dst = work / "train_gt.parquet"
    if not gt_dst.exists():
        gt = pl.read_csv(gt_src, separator="\t", quote_char=None,
                         schema_overrides={"source1_entity_id": pl.Utf8,
                                           "matched_entity_ids": pl.Utf8})
        gt = gt.with_columns(
            pl.col("matched_entity_ids").fill_null("").alias("m"))
        gt = gt.with_columns(
            pl.when(pl.col("m") == "")
              .then(pl.lit([], dtype=pl.List(pl.Utf8)))
              .otherwise(pl.col("m").str.split(","))
              .alias("matches")).drop("m", "matched_entity_ids")
        gt.write_parquet(gt_dst, compression="zstd")
        print(f"train_gt.parquet: {gt.height:,} rows")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
