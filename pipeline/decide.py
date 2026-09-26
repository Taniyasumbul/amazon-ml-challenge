import polars as pl


def _thresholds(cfg):
    if "thresholds" in cfg:
        values = cfg["thresholds"]
        return float(values["S2"]), float(values["S3"])
    if "t2" in cfg or "t3" in cfg:
        return float(cfg.get("t2", cfg.get("t3"))), float(cfg.get("t3", cfg.get("t2")))
    if "t" in cfg:
        return float(cfg["t"]), float(cfg["t"])
    raise ValueError("decision config must define thresholds for S2 and S3")


def decide(d, cfg):
    """Apply the configured threshold, parent filter, and per-source caps."""
    if cfg.get("mode", "threshold") != "threshold":
        raise ValueError("only threshold decision configs are supported")

    t2, t3 = _thresholds(cfg)
    min_score = float(cfg.get("min_score", 0.01))
    d = d.with_columns(
        pl.when(pl.col("is_s3") == 1).then(t3).otherwise(t2).alias("_threshold"))
    d = d.filter((pl.col("p") >= min_score) & (pl.col("p") >= pl.col("_threshold")))

    if cfg.get("one_parent", True):
        # Sort before deduplication so exact score ties resolve consistently.
        d = (d.sort(["b", "p", "a"], descending=[False, True, False])
               .unique(subset="b", keep="first", maintain_order=True))

    caps = cfg.get("caps", {"S2": 5, "S3": 6})
    if caps:
        k2, k3 = caps.get("S2"), caps.get("S3")
        d = (d.sort(["a", "is_s3", "p", "b"],
                    descending=[False, False, True, False])
               .with_columns(pl.col("p").rank("ordinal", descending=True)
                             .over(["a", "is_s3"]).alias("_rank"))
               .filter(pl.when(pl.col("is_s3") == 1)
                       .then(pl.col("_rank") <= k3 if k3 is not None else pl.lit(True))
                       .otherwise(pl.col("_rank") <= k2 if k2 is not None else pl.lit(True))))

    return d.select("a", "b")
