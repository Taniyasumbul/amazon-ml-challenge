import polars as pl

def prefilter(d):
    """One-parent rule (each S2/S3 record -> its best S1) + drop negligible probabilities."""
    return d.filter((pl.col("p") >= pl.col("p").max().over("b")) & (pl.col("p") >= 0.01))

def decide(d, cfg):
    d = prefilter(d)
    d = (d.with_columns(pl.col("p").rank("ordinal", descending=True).over(["a", "is_s3"]).alias("rk_s"))
          .filter(pl.col("rk_s") <= pl.when(pl.col("is_s3") == 1).then(6).otherwise(5)))
    if cfg["mode"] == "threshold":
        thr = pl.when(pl.col("is_s3") == 1).then(cfg["t3"]).otherwise(cfg["t2"])
        return d.filter(pl.col("p") >= thr).select("a", "b")
    # expected-F0.5 rule, chosen per S1
    pc = pl.col("p").clip(1e-6, 1 - 1e-6)
    d = d.with_columns(pl.col("p").sum().over("a").alias("T"),
                       (1 - pc).log().sum().over("a").exp().alias("F0"))
    d = d.filter(pl.col("p") >= cfg["t_low"])
    d = d.with_columns(pl.col("p").rank("ordinal", descending=True).over("a").alias("m"))
    d = d.sort(["a", "m"]).with_columns(pl.col("p").cum_sum().over("a").alias("E"))
    d = d.with_columns((1.25 * pl.col("E") / (pl.col("m") + 0.25 * pl.col("T"))).alias("F"))
    d = d.with_columns(pl.col("F").max().over("a").alias("Fmax"))
    d = d.with_columns(pl.when(pl.col("F") == pl.col("Fmax")).then(pl.col("m"))
                         .otherwise(None).min().over("a").alias("mstar"))
    return (d.filter((pl.col("m") <= pl.col("mstar")) & (pl.col("Fmax") > pl.col("F0") * cfg["k0"]))
             .select("a", "b"))
