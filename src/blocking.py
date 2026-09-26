"""Stage 2: candidate generation.

Builds an inverted index over blocking keys and returns, for every Source 1
record, the top-K Source 2 / Source 3 records ranked by IDF-weighted shared-key
score.  Run per country, chunked over Source 1, so peak memory stays bounded.

PYTHONHASHSEED must be fixed (the runner sets it to 0) so key hashes are
reproducible across processes.
"""
from __future__ import annotations

import math
import numpy as np
import polars as pl

from keys import record_keys, N_FAM

KEY_CHUNK = 400_000       # records per key-building chunk
S1_CHUNK = 150_000        # Source-1 records per join chunk
MAX_KEY_DF = 120          # a key matching more than this many targets is dropped

# Ranking weights over the per-family IDF evidence, fitted on training data by
# fit_rank_weights.py.  Uniform weights are the fallback.
RANK_WEIGHTS = [1.0] * N_FAM

def token_df(df: pl.DataFrame, col: str, is_list: bool) -> dict:
    """Document frequency of each token over a frame."""
    if is_list:
        s = df.select(pl.col(col).list.unique().alias("t")).explode("t")
    else:
        s = df.select(pl.col(col).str.split(" ").list.unique().alias("t")).explode("t")
    vc = s.drop_nulls().filter(pl.col("t").str.len_chars() > 1)["t"].value_counts()
    return dict(zip(vc["t"].to_list(), vc["count"].to_list()))


def build_keys(df: pl.DataFrame, name_df: dict, addr_df: dict):
    """Return (row_idx int32, key_hash uint64, family uint8) arrays."""
    n = df.height
    idx_a, hsh_a, fam_a = [], [], []
    for start in range(0, n, KEY_CHUNK):
        end = min(start + KEY_CHUNK, n)
        sub = df.slice(start, end - start)
        names = sub["name_core"].to_list()
        atoks = sub["addr_toks"].to_list()
        anums = sub["addr_nums"].to_list()
        idx, hsh, fam = [], [], []
        for i in range(end - start):
            ri = start + i
            for f, k in record_keys(names[i], atoks[i] or [], anums[i] or [],
                                    name_df, addr_df):
                idx.append(ri)
                hsh.append(hash(k) & 0xFFFFFFFFFFFFFFFF)
                fam.append(f)
        idx_a.append(np.asarray(idx, dtype=np.int32))
        hsh_a.append(np.asarray(hsh, dtype=np.uint64))
        fam_a.append(np.asarray(fam, dtype=np.uint8))
    if not idx_a:
        return (np.empty(0, np.int32), np.empty(0, np.uint64), np.empty(0, np.uint8))
    return (np.concatenate(idx_a), np.concatenate(hsh_a), np.concatenate(fam_a))


def keys_frame(df: pl.DataFrame, name_df: dict, addr_df: dict, side: str) -> pl.DataFrame:
    idx, hsh, fam = build_keys(df, name_df, addr_df)
    return pl.DataFrame({f"i{side}": idx, "h": hsh, f"fam{side}": fam}).unique()


def candidates(s1: pl.DataFrame, tgt: pl.DataFrame, name_df: dict, addr_df: dict,
               top_k: int, max_key_df: int = MAX_KEY_DF,
               weights=None, log=print) -> pl.DataFrame:
    """Top-K candidates from `tgt` for each row of `s1` (both single-country)."""
    k1 = keys_frame(s1, name_df, addr_df, "1")
    kt = keys_frame(tgt, name_df, addr_df, "t")
    log(f"    keys: s1={k1.height:,} tgt={kt.height:,}")

    # drop keys that are not selective on the target side
    kdf = kt.group_by("h").len().rename({"len": "kdf"})
    kdf = kdf.filter(pl.col("kdf") <= max_key_df)
    kt = kt.join(kdf, on="h", how="inner")
    k1 = k1.join(kdf.select("h"), on="h", how="inner")
    log(f"    after df cap: s1={k1.height:,} tgt={kt.height:,}")

    w = RANK_WEIGHTS if weights is None else weights
    ntgt = max(tgt.height, 2)
    kt = kt.with_columns(
        (np.log(ntgt) - pl.col("kdf").cast(pl.Float32).log()).cast(pl.Float32).alias("w"))

    out = []
    n1 = s1.height
    for start in range(0, n1, S1_CHUNK):
        end = min(start + S1_CHUNK, n1)
        part = k1.filter((pl.col("i1") >= start) & (pl.col("i1") < end))
        if part.height == 0:
            continue
        j = part.join(kt, on="h", how="inner")
        if j.height == 0:
            continue
        # Per-family IDF evidence.  Keeping the families separate lets the
        # ranker learn how much each one is worth instead of assuming they are
        # interchangeable.
        agg = j.group_by(["i1", "it"]).agg(
            [pl.when(pl.col("famt") == f).then(pl.col("w")).otherwise(0.0)
               .sum().cast(pl.Float32).alias(f"f{f}") for f in range(N_FAM)]
            + [pl.col("famt").n_unique().cast(pl.UInt8).alias("nfam")]
        )
        agg = agg.with_columns(
            sum(pl.col(f"f{f}") * float(w[f]) for f in range(N_FAM))
            .cast(pl.Float32).alias("bscore"))
        agg = (agg.sort(["i1", "bscore"], descending=[False, True])
                  .group_by("i1", maintain_order=True).head(top_k))
        out.append(agg)
        log(f"    chunk {start:,}-{end:,}: join={j.height:,} kept={agg.height:,}")
        del j, part
    if not out:
        sch = {"i1": pl.Int32, "it": pl.Int32}
        sch.update({f"f{f}": pl.Float32 for f in range(N_FAM)})
        sch.update({"nfam": pl.UInt8, "bscore": pl.Float32})
        return pl.DataFrame(schema=sch)
    return pl.concat(out)
