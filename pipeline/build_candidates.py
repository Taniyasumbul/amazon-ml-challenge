"""Build ranked candidate pairs for one split, country, and target source."""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import blocking  # noqa: E402
from keys import N_FAM  # noqa: E402

blocking.KEY_CHUNK = 100_000
keys_frame = blocking.keys_frame
token_df = blocking.token_df

KEY_CHUNK = 100_000
S1_CHUNK = 25_000
PIECE = 500_000
TOPK_RAW = 300
MAX_KEY_DF = 120
RECORD_COLS = ["entity_id", "country", "name_core", "addr_toks", "addr_nums"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("train", "test"))
    parser.add_argument("country")
    parser.add_argument("source", choices=("2", "3"))
    parser.add_argument("top_k", type=int)
    parser.add_argument("--ids-file", help="optional parquet with sampled entity_id values")
    parser.add_argument(
        "--norm-dir",
        default=os.environ.get("AMAZON_NORM_DIR", "/content/drive/MyDrive/amazon_norm"),
    )
    parser.add_argument("--model", default=str(ROOT / "models/block_ranker_lgb.txt"))
    parser.add_argument("--overwrite", action="store_true", help="rebuild existing chunks")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("top_k must be positive")
    if args.split == "test" and args.ids_file:
        parser.error("--ids-file is only valid for the train split")
    if os.environ.get("PYTHONHASHSEED") != "0":
        parser.error("run with PYTHONHASHSEED=0 so blocking-key hashes stay reproducible")
    return args


def main():
    args = parse_args()
    norm = Path(args.norm_dir)
    out_dir = norm / "blocks"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def log(message):
        print(f"{message} | {time.time() - started:.0f}s", flush=True)

    def load(path):
        return (pl.scan_parquet(path).filter(pl.col("country") == args.country)
                .select(RECORD_COLS).collect())

    def chunk_path(index):
        return out_dir / f"{args.split}_{args.country}_S{args.source}_{index:03d}.parquet"

    if not (norm / f"{args.split}_s1.parquet").exists():
        raise FileNotFoundError(f"run src/prep.py first; missing {norm}/{args.split}_s1.parquet")
    if not (norm / f"{args.split}_s{args.source}.parquet").exists():
        raise FileNotFoundError(f"missing {norm}/{args.split}_s{args.source}.parquet")
    if not Path(args.model).exists():
        raise FileNotFoundError(f"blocking ranker model not found: {args.model}")

    s1_all = load(norm / f"{args.split}_s1.parquet")
    s1 = s1_all
    if args.ids_file:
        ids = pl.read_parquet(args.ids_file, columns=["entity_id"]).unique()
        s1 = s1_all.join(ids, on="entity_id", how="semi")
    n_chunks = (s1.height + S1_CHUNK - 1) // S1_CHUNK
    todo = [i for i in range(n_chunks) if args.overwrite or not chunk_path(i).exists()]
    log(f"{args.split} {args.country} S{args.source} K={args.top_k}: "
        f"S1={s1.height:,}, chunks={n_chunks}, todo={len(todo)}")
    if not todo:
        return

    target = load(norm / f"{args.split}_s{args.source}.parquet")
    union = pl.concat([s1_all.select("name_core", "addr_toks"),
                       target.select("name_core", "addr_toks")])
    name_df = token_df(union, "name_core", False)
    addr_df = token_df(union, "addr_toks", True)
    del union, s1_all
    gc.collect()

    target_ids = pl.DataFrame({"it": np.arange(target.height, dtype=np.int32),
                               "b": target["entity_id"]})
    target_key_parts = []
    for start in range(0, target.height, PIECE):
        keys = keys_frame(target.slice(start, PIECE), name_df, addr_df, "t")
        target_key_parts.append(keys.with_columns((pl.col("it") + start).cast(pl.Int32)))
        del keys
        gc.collect()
    if not target_key_parts:
        log("target source is empty; no candidates generated")
        return
    target_keys = pl.concat(target_key_parts)
    del target_key_parts, target
    gc.collect()

    key_df = target_keys.group_by("h").len().filter(pl.col("len") <= MAX_KEY_DF)
    target_keys = (target_keys.join(key_df, on="h", how="inner")
                   .with_columns((np.log(max(target_ids.height, 2))
                                  - pl.col("len").cast(pl.Float32).log())
                                 .cast(pl.Float32).alias("w"))
                   .drop("len"))
    keep_hashes = key_df.select("h")
    del key_df
    gc.collect()

    model = lgb.Booster(model_file=args.model)
    ranker_features = (
        [f"f{i}" for i in range(N_FAM)] + ["nfam", "bscore", "b_rel", "b_rank", "n_cand"]
        + [f"f{i}_rel" for i in range(N_FAM)]
    )

    for index in todo:
        chunk = s1.slice(index * S1_CHUNK, S1_CHUNK)
        s1_keys = keys_frame(chunk, name_df, addr_df, "1").join(
            keep_hashes, on="h", how="semi")
        joined = s1_keys.join(target_keys, on="h", how="inner")
        del s1_keys
        if not joined.height:
            pl.DataFrame(schema={"a": pl.Utf8, "b": pl.Utf8}).write_parquet(chunk_path(index))
            log(f"chunk {index}: no shared blocking keys")
            continue

        aggregate = joined.group_by(["i1", "it"]).agg(
            [pl.when(pl.col("famt") == family).then(pl.col("w")).otherwise(0.0)
             .sum().cast(pl.Float32).alias(f"f{family}") for family in range(N_FAM)]
            + [pl.col("famt").n_unique().cast(pl.UInt8).alias("nfam")]
        )
        del joined
        gc.collect()
        aggregate = aggregate.with_columns(
            pl.sum_horizontal([f"f{i}" for i in range(N_FAM)])
            .cast(pl.Float32).alias("bscore"))
        aggregate = (aggregate.sort(["i1", "bscore"], descending=[False, True])
                     .group_by("i1", maintain_order=True).head(TOPK_RAW))

        relative_features = [
            (pl.col(f"f{i}") / (pl.col(f"f{i}").max().over("i1") + 1e-6))
            .alias(f"f{i}_rel") for i in range(N_FAM)
        ]
        aggregate = aggregate.with_columns(
            (pl.col("bscore") / pl.col("bscore").max().over("i1")).alias("b_rel"),
            pl.col("bscore").rank("ordinal", descending=True).over("i1")
            .cast(pl.Float32).alias("b_rank"),
            pl.len().over("i1").cast(pl.Float32).alias("n_cand"),
            *relative_features,
        )
        scores = model.predict(aggregate.select(ranker_features).to_numpy().astype(np.float32))
        aggregate = aggregate.with_columns(pl.Series("rscore", scores, dtype=pl.Float32))
        aggregate = (aggregate.with_columns(
            pl.col("rscore").rank("ordinal", descending=True).over("i1").alias("rrank"))
            .filter(pl.col("rrank") <= args.top_k))

        s1_map = (chunk.select(pl.col("entity_id").alias("a")).with_row_index("i1")
                  .with_columns(pl.col("i1").cast(pl.Int32)))
        aggregate = (aggregate.with_columns(pl.col("i1").cast(pl.Int32),
                                            pl.col("it").cast(pl.Int32))
                     .join(s1_map, on="i1").join(target_ids, on="it")
                     .select(["a", "b"] + [f"f{i}" for i in range(N_FAM)]
                             + ["nfam", "bscore", "b_rel", "rscore", "rrank"]))
        aggregate.write_parquet(chunk_path(index))
        log(f"chunk {index}: {chunk.height:,} S1 -> {aggregate.height:,} pairs")
        del aggregate, chunk
        gc.collect()

    log("candidate generation complete")


if __name__ == "__main__":
    main()
