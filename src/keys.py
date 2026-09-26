"""Blocking key generation.

A record emits a small set of *exact* keys; two records become candidates when
they share at least one key.  Complementary families keep recall high when any
single field is destroyed by noise:

  n_full   sorted core name tokens             clean name, any word order
  n_tok    a rare core name token              survives typos elsewhere
  n_pair   two rare core name tokens           selective, survives one typo
  n_skel   consonant skeleton of the name      survives transliteration drift
  n_sktok  consonant skeleton of a rare token  ditto, single-token evidence
  n_anag   sorted characters of the name       survives word order + concatenation
  na       rare name token + address token     survives noise in either field
  a_pair   two rare address tokens             survives a totally different name
  a_num    house number + address token        numeric anchor

Token choice is rarity-driven: document frequency is computed over the union of
the sources for that country, and only a record's rarest tokens feed the
name-side families.  Address families deliberately mix the rarest tokens with
the *commonest* ones (city / state), because truncated Source 2 and Source 3
addresses often retain nothing else.

The skeleton and anagram families exist because Source 2 and Source 3 render
Indic names phonetically ("golden services" -> "goldan sarvisas") and sometimes
collapse a name into a domain ("Aditraj Agro" -> "agroaditraj.com").
"""
from __future__ import annotations

NAME_STOP = {"and", "the", "of", "for", "in", "at", "de", "du", "des", "la",
             "le", "les", "el", "a", "an", "s", "india", "france", "com",
             "www", "net", "org", "co"}

FAMILIES = ["n_full", "n_tok", "n_pair", "na", "a_pair", "a_num",
            "n_skel", "n_sktok", "n_anag"]
FAM_ID = {f: i for i, f in enumerate(FAMILIES)}
N_FAM = len(FAMILIES)

N_NAME_TOK = 3        # rarest name tokens used by n_tok / n_pair
N_ADDR_RARE = 3       # rarest address tokens
N_ADDR_COMMON = 2     # commonest address tokens (city / state anchors)
N_NUM = 2             # house numbers used by a_num
NA_NAME = 2           # name tokens used by the cross family
MAX_DF_TOK = 400_000  # ignore hopelessly common tokens entirely

# --- consonant skeleton ----------------------------------------------------
# Digraphs first, then vowel deletion and consonant folding.  The folds merge
# the sound pairs that Indic->Latin transliteration swaps: ph/f, bh/v/b/w,
# c/k/q, s/z/sh, and drops y.
_DIGRAPHS = (("chh", "c"), ("sch", "s"), ("sh", "s"), ("ph", "f"), ("bh", "v"),
             ("gh", "g"), ("dh", "d"), ("th", "t"), ("kh", "k"), ("ck", "k"),
             ("ch", "c"), ("wh", "v"), ("gn", "n"), ("ps", "s"))
_FOLD = {"w": "v", "b": "v", "q": "k", "k": "k", "c": "s", "z": "s",
         "x": "ks", "y": "", "j": "j"}
_VOWELS = frozenset("aeiou")


def skeleton(tok: str) -> str:
    s = tok
    for a, b in _DIGRAPHS:
        if a in s:
            s = s.replace(a, b)
    out = []
    for ch in s:
        if ch in _VOWELS or ch.isdigit():
            continue
        ch = _FOLD.get(ch, ch)
        if ch and (not out or out[-1] != ch):
            out.append(ch)
    return "".join(out)


def _by_df(toks, df, max_df=MAX_DF_TOK):
    seen = {}
    for t in toks:
        if t not in seen:
            d = df.get(t, 1)
            if d <= max_df:
                seen[t] = d
    return sorted(seen, key=lambda t: (seen[t], t))


def record_keys(name_core, addr_toks, addr_nums, name_df, addr_df):
    """Yield (family_id, key_string) for one record."""
    nt = [t for t in name_core.split() if len(t) > 1 and t not in NAME_STOP]
    at = [t for t in addr_toks if len(t) > 1 and not t.isdigit()]
    nums = []
    for x in addr_nums:
        x = x.lstrip("0") or "0"
        if 1 <= len(x) <= 7 and x not in nums:
            nums.append(x)

    n_sorted = _by_df(nt, name_df)
    n_rare = n_sorted[:N_NAME_TOK]
    a_sorted = _by_df(at, addr_df)
    a_rare = a_sorted[:N_ADDR_RARE]
    a_common = [t for t in a_sorted[-N_ADDR_COMMON:] if t not in a_rare]
    a_mix = a_rare[:2] + a_common

    if nt:
        uniq = sorted(set(nt))
        yield 0, " ".join(uniq)                                   # n_full
        cat = "".join(uniq)
        sk = skeleton(cat)
        if len(sk) >= 4:
            yield 6, sk                                           # n_skel
        if len(cat) >= 8:
            yield 8, "".join(sorted(cat))                         # n_anag
    for t in n_rare:
        yield 1, t                                                # n_tok
        sk = skeleton(t)
        if len(sk) >= 4:
            yield 7, sk                                           # n_sktok
    for i in range(len(n_rare)):                                  # n_pair
        for j in range(i + 1, len(n_rare)):
            a, b = sorted((n_rare[i], n_rare[j]))
            yield 2, a + "|" + b
    for t in n_rare[:NA_NAME]:                                    # na
        for a in a_mix:
            yield 3, t + "|" + a
    for i in range(len(a_rare)):                                  # a_pair
        for j in range(i + 1, len(a_rare)):
            a, b = sorted((a_rare[i], a_rare[j]))
            yield 4, a + "|" + b
    for n in nums[:N_NUM]:                                        # a_num
        for a in a_mix:
            yield 5, n + "|" + a
