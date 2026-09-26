import numpy as np, polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

REC_COLS = ["entity_id", "name_norm", "name_core", "name_alias",
            "addr_norm", "addr_toks", "addr_nums"]

def _nz(e):
    return pl.when(e == "").then(None).otherwise(e)

def prep(df, sfx):
    nc = pl.col("name_core").fill_null("")
    nums = (pl.col("addr_nums").list.eval(pl.element().str.strip_chars_start("0"))
              .list.eval(pl.element().filter(pl.element() != "")).list.unique())
    out = df.select(
        pl.col("entity_id").alias("id"),
        pl.col("name_norm").fill_null("").alias("nn"),
        nc.alias("nc"),
        pl.col("name_alias").fill_null("").alias("al"),
        pl.col("addr_norm").fill_null("").alias("ad"),
        nc.str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().alias("nt"),
        pl.col("addr_toks").list.unique().alias("at"),
        pl.col("addr_toks").list.tail(2).alias("tl"),
        nums.alias("an"),
        _nz(pl.col("addr_nums").list.first().str.strip_chars_start("0")).alias("n0"),
        nc.str.replace_all(" ", "").alias("ns"),
    )
    out = out.with_columns(
        pl.col("an").list.eval(pl.element().filter(
            pl.element().str.len_chars().is_between(5, 6))).alias("zp"),
        pl.col("ns").str.split("").list.sort().list.join("").alias("ag"),
        pl.col("nc").str.len_chars().alias("nl"),
    )
    return out.rename({c: f"{c}_{sfx}" for c in out.columns})

STR = [("n_ratio", "nc", fuzz.ratio), ("n_tsort", "nc", fuzz.token_sort_ratio),
       ("n_tset", "nc", fuzz.token_set_ratio), ("n_partial", "nc", fuzz.partial_ratio),
       ("n_jw", "nc", JaroWinkler.normalized_similarity), ("n_nosp", "ns", fuzz.ratio),
       ("nn_ratio", "nn", fuzz.ratio), ("a_ratio", "ad", fuzz.ratio),
       ("a_tset", "ad", fuzz.token_set_ratio)]

BLOCK = [f"f{i}" for i in range(9)] + ["nfam", "bscore", "b_rel", "rscore", "rrank"]
FEATS = ([n for n, _, _ in STR] + ["alias_x", "n_jac", "n_contain", "a_jac", "num_jac",
         "num_shared", "tail_shared", "zip_shared", "n_eq", "nn_eq", "ag_eq", "n_lenr",
         "n0_eq", "n0_suffix", "a_empty_any", "a_empty_both", "n_empty_any",
         "n_cand", "r_gap"] + BLOCK + ["is_s3"])

def _cp(x, y, sc):
    return cpdist(x, y, scorer=sc, workers=-1, dtype=np.float32)

def _inter(c):
    return pl.col(f"{c}_a").list.set_intersection(pl.col(f"{c}_b")).list.len()

def _jac(i, x, y):
    u = pl.col(x) + pl.col(y) - pl.col(i)
    return pl.when(u > 0).then(pl.col(i) / u).otherwise(0.0)

def pair_features(p, A, B, is_s3):
    d = (p.join(A, left_on="a", right_on="id_a", how="left")
          .join(B, left_on="b", right_on="id_b", how="left"))
    cols = {}
    for name, c, sc in STR:
        cols[name] = _cp(d[f"{c}_a"].fill_null("").to_list(),
                         d[f"{c}_b"].fill_null("").to_list(), sc)
    al1 = _cp(d["al_b"].fill_null("").to_list(), d["nc_a"].fill_null("").to_list(), fuzz.token_set_ratio)
    al2 = _cp(d["al_a"].fill_null("").to_list(), d["nc_b"].fill_null("").to_list(), fuzz.token_set_ratio)
    cols["alias_x"] = np.maximum(al1, al2)
    d = d.with_columns([pl.Series(k, v) for k, v in cols.items()])

    d = d.with_columns(
        _inter("nt").alias("_nti"), pl.col("nt_a").list.len().alias("_nta"),
        pl.col("nt_b").list.len().alias("_ntb"),
        _inter("at").alias("_ati"), pl.col("at_a").list.len().alias("_ata"),
        pl.col("at_b").list.len().alias("_atb"),
        _inter("an").alias("num_shared"), pl.col("an_a").list.len().alias("_ana"),
        pl.col("an_b").list.len().alias("_anb"),
        _inter("tl").alias("tail_shared"), _inter("zp").alias("zip_shared"),
    )
    mn = pl.min_horizontal("_nta", "_ntb")
    d = d.with_columns(
        _jac("_nti", "_nta", "_ntb").alias("n_jac"),
        pl.when(mn > 0).then(pl.col("_nti") / mn).otherwise(0.0).alias("n_contain"),
        _jac("_ati", "_ata", "_atb").alias("a_jac"),
        _jac("num_shared", "_ana", "_anb").alias("num_jac"),
        ((pl.col("nc_a") == pl.col("nc_b")) & (pl.col("nc_a") != "")).alias("n_eq"),
        ((pl.col("nn_a") == pl.col("nn_b")) & (pl.col("nn_a") != "")).alias("nn_eq"),
        ((pl.col("ag_a") == pl.col("ag_b")) & (pl.col("ag_a") != "")).alias("ag_eq"),
        (pl.min_horizontal("nl_a", "nl_b") / pl.max_horizontal("nl_a", "nl_b")).alias("n_lenr"),
        (pl.col("n0_a") == pl.col("n0_b")).alias("n0_eq"),
        ((pl.col("n0_a").str.ends_with(pl.col("n0_b")) | pl.col("n0_b").str.ends_with(pl.col("n0_a")))
         & (pl.col("n0_a") != pl.col("n0_b"))).alias("n0_suffix"),
        ((pl.col("ad_a") == "") | (pl.col("ad_b") == "")).alias("a_empty_any"),
        ((pl.col("ad_a") == "") & (pl.col("ad_b") == "")).alias("a_empty_both"),
        ((pl.col("nc_a") == "") | (pl.col("nc_b") == "")).alias("n_empty_any"),
        pl.len().over("a").alias("n_cand"),
        (pl.col("rscore") - pl.col("rscore").max().over("a")).alias("r_gap"),
        pl.lit(1.0 if is_s3 else 0.0).alias("is_s3"),
    )
    return d.select(["a", "b"] + [pl.col(f).cast(pl.Float32).fill_null(0).fill_nan(0)
                                  for f in FEATS])
