"""Field normalisation for business records.

Everything here is derived from the provided data or from static *linguistic*
tables (script transliteration, street-type abbreviations, legal-form words).
No external database, API or geographic lookup is used.
"""
from __future__ import annotations

import re
import unicodedata

import polars as pl

from translit import has_indic, transliterate

# ---------------------------------------------------------------------------
# Token tables
# ---------------------------------------------------------------------------

# Legal form / company-type words.  Highly frequent and almost never
# discriminative, so they are stripped from the "core" name.
LEGAL_TOKENS = {
    # anglophone
    "llc", "lc", "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "lp", "llp", "pc", "pllc", "plc", "pa", "ltda",
    "trust", "incorporation", "corporated",
    # India
    "pvt", "private", "opc", "praivet", "praivate", "pra", "li", "limitd",
    "kampani", "kampni", "prai",
    # France
    "sarl", "sarlu", "sas", "sasu", "sa", "eurl", "sci", "snc", "scp", "scm",
    "selarl", "selas", "scop", "gie", "sca", "scs", "scic", "sem", "cie",
    "ets", "etablissements", "etablissement", "sarls", "eirl", "scea", "gaec",
}

# Honorifics and title noise that appear as name prefixes.
HONORIFIC_TOKENS = {
    "mr", "mrs", "ms", "smt", "dr", "shri", "sri", "sh", "m", "messrs",
    "miss", "prof", "late",
}

# "doing business as" style markers - the segment *after* the marker is the
# real trading name (verified against the training clusters).
DBA_RE = re.compile(
    r"\b(?:d\s*b\s*a|dba|doing\s+business\s+as|trading\s+as|t\s*/\s*a|"
    r"formerly\s+known\s+as|formerly|f\s*/\s*k\s*/\s*a|fka|also\s+known\s+as|aka|"
    r"now\s+known\s+as|nka)\b"
)

# Street-type and address abbreviations, canonicalised to the long form that
# Source 1 tends to use.
ADDR_ABBREV = {
    # US street types
    "st": "street", "str": "street", "rd": "road", "dr": "drive", "drv": "drive",
    "ave": "avenue", "aven": "avenue", "av": "avenue", "ln": "lane",
    "ct": "court", "cir": "circle", "crcl": "circle", "blvd": "boulevard",
    "blv": "boulevard", "pl": "place", "hwy": "highway", "pkwy": "parkway",
    "pky": "parkway", "sq": "square", "ter": "terrace", "terr": "terrace",
    "trl": "trail", "trc": "trace", "pt": "point", "cv": "cove", "bnd": "bend",
    "xing": "crossing", "holw": "hollow", "mnr": "manor", "plz": "plaza",
    "spg": "spring", "spgs": "springs", "byp": "bypass", "cswy": "causeway",
    "expy": "expressway", "fwy": "freeway", "jct": "junction", "lk": "lake",
    "mt": "mount", "mtn": "mountain", "vlg": "village", "vly": "valley",
    "crk": "creek", "frst": "forest", "grn": "green", "hts": "heights",
    "is": "island", "knl": "knoll", "mdw": "meadow", "orch": "orchard",
    "pne": "pine", "rdg": "ridge", "riv": "river", "shr": "shore",
    "stn": "station", "vw": "view", "wy": "way", "gdn": "garden",
    "gdns": "gardens", "clb": "club", "cyn": "canyon", "hl": "hill",
    "hls": "hills", "psge": "passage", "loop": "loop", "run": "run",
    # unit / building
    "apt": "apartment", "aptt": "apartment", "ste": "suite", "bldg": "building",
    "bld": "building", "fl": "floor", "flr": "floor", "rm": "room",
    "dept": "department", "bsmt": "basement", "frnt": "front", "ofc": "office",
    # directions
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    # France
    "r": "rue", "bd": "boulevard", "bld": "boulevard", "bvd": "boulevard",
    "imp": "impasse", "all": "allee", "che": "chemin", "ch": "chemin",
    "rte": "route", "res": "residence", "bat": "batiment", "sq": "square",
    "pas": "passage", "qu": "quai", "fbg": "faubourg", "crs": "cours",
    "vla": "villa", "sen": "sentier", "esp": "esplanade", "prom": "promenade",
    # India
    "opp": "opposite", "nr": "near", "marg": "road", "sec": "sector",
    "blk": "block", "gr": "ground", "colly": "colony", "extn": "extension",
    "ext": "extension", "mkt": "market", "indl": "industrial",
    "bldng": "building", "hno": "house", "ho": "house",
}

# Tokens that carry no matching signal in an address.
ADDR_STOP = {
    "no", "number", "the", "of", "at", "de", "du", "des", "la", "le", "les",
    "a", "and", "et", "near", "opposite", "behind", "above", "below", "po",
    "box", "door", "plot", "flat", "shop", "unit", "house", "building",
    "floor", "room", "suite", "apartment", "block",
}

NULLISH = {"", "null", "nan", "none", "n/a", "na", "-", "--", "unknown"}

_COMBINING = re.compile(r"[̀-ͯ᪰-᫿⃐-⃰]")
_NONWORD = re.compile(r"[^0-9a-z]+")
_SINGLE_RUN = re.compile(r"\b(?:[a-z] ){1,}[a-z]\b")
_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------

def ascii_fold(s: str) -> str:
    """Strip accents/diacritics, keeping the base Latin letter."""
    return _COMBINING.sub("", unicodedata.normalize("NFKD", s))


def _merge_letter_runs(s: str) -> str:
    """'l l c' -> 'llc', 's a r l' -> 'sarl' (initialisms lost to punctuation)."""
    return _SINGLE_RUN.sub(lambda m: m.group(0).replace(" ", ""), s)


def basic_clean(s: str) -> str:
    if not s:
        return ""
    if has_indic(s):
        s = transliterate(s)
    s = ascii_fold(s)
    s = s.lower().replace("&", " and ")
    s = _NONWORD.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    if s in NULLISH:
        return ""
    return _merge_letter_runs(s)


def clean_name(s: str) -> str:
    """Normalise a business name, resolving DBA/formerly aliases."""
    if not s:
        return ""
    if has_indic(s):
        s = transliterate(s)
    s = ascii_fold(s).lower().replace("&", " and ")
    s = _NONWORD.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    if s in NULLISH:
        return ""
    # keep only the segment after the last DBA marker
    parts = DBA_RE.split(s)
    if len(parts) > 1 and parts[-1].strip():
        s = parts[-1].strip()
    return _merge_letter_runs(s)


def name_alias(s: str) -> str:
    """The segment *before* a DBA marker (the discarded alias), if any."""
    if not s:
        return ""
    if has_indic(s):
        s = transliterate(s)
    s = ascii_fold(s).lower().replace("&", " and ")
    s = _NONWORD.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    parts = DBA_RE.split(s)
    return _merge_letter_runs(parts[0].strip()) if len(parts) > 1 else ""


def core_tokens(name_norm: str) -> list[str]:
    """Name tokens with legal forms, honorifics and bare digits removed."""
    toks = [t for t in name_norm.split()
            if t not in LEGAL_TOKENS and t not in HONORIFIC_TOKENS]
    # drop leading honorific-only residue but never return nothing
    return toks if toks else name_norm.split()


def addr_tokens(addr_norm: str) -> list[str]:
    out = []
    for t in addr_norm.split():
        t = ADDR_ABBREV.get(t, t)
        if t in ADDR_STOP:
            continue
        out.append(t)
    return out


_NUM = re.compile(r"\d+")


def addr_numbers(addr_raw_norm: str) -> list[str]:
    """Digit runs in an address: house numbers, PIN codes, sector numbers."""
    return _NUM.findall(addr_raw_norm)


# ---------------------------------------------------------------------------
# Frame-level entry point
# ---------------------------------------------------------------------------

def normalize_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Add normalised columns to a raw source frame."""
    names = df["business_name"].fill_null("").to_list()
    addrs = df["business_address"].fill_null("").to_list()

    name_norm = [clean_name(s) for s in names]
    alias = [name_alias(s) for s in names]
    addr_norm = [basic_clean(s) for s in addrs]

    out = df.with_columns([
        pl.Series("name_norm", name_norm, dtype=pl.Utf8),
        pl.Series("name_alias", alias, dtype=pl.Utf8),
        pl.Series("addr_norm", addr_norm, dtype=pl.Utf8),
    ])
    out = out.with_columns([
        pl.Series("name_core", [" ".join(core_tokens(s)) for s in name_norm], dtype=pl.Utf8),
        pl.Series("addr_toks", [addr_tokens(s) for s in addr_norm], dtype=pl.List(pl.Utf8)),
        pl.Series("addr_nums", [addr_numbers(s) for s in addr_norm], dtype=pl.List(pl.Utf8)),
    ])
    return out
