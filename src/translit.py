"""Rule-based Indic -> Latin transliteration.

All nine Indic scripts in this dataset (Devanagari, Bengali, Gurmukhi, Gujarati,
Oriya, Tamil, Telugu, Kannada, Malayalam) are code-point aligned: the same
phonetic slot sits at the same offset inside each 128-point block.  So we fold
every block onto Devanagari and then apply a single Devanagari -> Latin table.

No external data is used - this is a static phonetic table.
"""
import re

# Start code point of each Indic block that is layout-aligned with Devanagari.
_BLOCKS = [0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]
_DEV = 0x0900

_FOLD = {}
for _b in _BLOCKS:
    for _o in range(0x80):
        _FOLD[chr(_b + _o)] = chr(_DEV + _o)

# Independent vowels
_VOWEL = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
    "ऋ": "ri", "ॠ": "ri", "ऌ": "li", "ॡ": "li",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "ऍ": "e", "ऎ": "e", "ऑ": "o", "ऒ": "o",
}
# Dependent vowel signs (matras).  "" == inherent 'a' suppressed by the sign.
_MATRA = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u",
    "ृ": "ri", "ॄ": "ri", "ॢ": "li", "ॣ": "li",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ॅ": "e", "ॆ": "e", "ॉ": "o", "ॊ": "o",
}
# Consonants carry an inherent 'a' unless a matra or virama follows.
_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "ऩ": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ऱ": "r", "ल": "l", "ळ": "l", "ऴ": "l",
    "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    # nukta forms
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y",
}
_SIGN = {
    "ं": "n",   # anusvara
    "ँ": "n",   # chandrabindu
    "ः": "h",   # visarga
    "ऽ": "",    # avagraha
    "ॐ": "om",
    "़": "",    # bare nukta
    "।": " ", "॥": " ",
}
_DIGIT = {chr(0x0966 + i): str(i) for i in range(10)}
_VIRAMA = "्"

_NUKTA_PAIRS = {"क़", "ख़", "ग़", "ज़", "ड़", "ढ़", "फ़", "य़"}
_HAS_INDIC = re.compile(r"[ऀ-෿]")


# --- Latin letter names spelled out in Indic script -------------------------
# Indian company names routinely spell acronyms phonetically: एलएलपी = "LLP",
# एसएस = "SS".  The generic rules render those as "elaelapi" / "esaes", so we
# detect whole words that decompose into a run of letter names and emit the
# acronym instead.
_LETTER_NAMES = {
    "\u090f\u0932": "l", "\u090f\u0938": "s", "\u090f\u092e": "m", "\u090f\u0928": "n",
    "\u090f\u092b": "f", "\u090f\u091a": "h", "\u090f\u0915\u094d\u0938": "x",
    "\u092c\u0940": "b", "\u0938\u0940": "c", "\u0921\u0940": "d", "\u091c\u0940": "g",
    "\u091c\u0947": "j", "\u0915\u0947": "k", "\u092a\u0940": "p", "\u091f\u0940": "t",
    "\u0935\u0940": "v", "\u092f\u0942": "u", "\u0906\u0930": "r", "\u0906\u0908": "i",
    "\u0908": "e", "\u0913": "o", "\u0915\u094d\u092f\u0942": "q",
    "\u091c\u0947\u0921": "z", "\u0921\u092c\u094d\u0932\u094d\u092f\u0942": "w",
    "\u0935\u093e\u0908": "y", "\u090f": "a",
}
_LN_MAX = max(len(k) for k in _LETTER_NAMES)


def _as_acronym(w):
    """Return the acronym if `w` is wholly a run of >=2 letter names, else None."""
    out, i, n = [], 0, len(w)
    while i < n:
        for ln in range(min(_LN_MAX, n - i), 0, -1):
            piece = _LETTER_NAMES.get(w[i:i + ln])
            if piece is not None:
                out.append(piece)
                i += ln
                break
        else:
            return None
    return "".join(out) if len(out) >= 2 else None


def has_indic(s: str) -> bool:
    return bool(_HAS_INDIC.search(s))


def _fold(s: str) -> str:
    return "".join(_FOLD.get(c, c) for c in s)


def translit_word(w: str) -> str:
    """Transliterate one folded-to-Devanagari word into Latin."""
    acro = _as_acronym(w)
    if acro is not None:
        return acro
    out = []
    i, n = 0, len(w)
    while i < n:
        c = w[i]
        # combine base consonant + nukta into its own phoneme when known
        if i + 1 < n and (c + w[i + 1]) in _NUKTA_PAIRS:
            c = c + w[i + 1]
            i += 1
        if c in _CONS:
            out.append(_CONS[c])
            nxt = w[i + 1] if i + 1 < n else ""
            if nxt == _VIRAMA:
                i += 2
                continue
            if nxt in _MATRA:
                out.append(_MATRA[nxt])
                i += 2
                continue
            # inherent 'a', except word-final (schwa deletion)
            if i + 1 < n:
                out.append("a")
            i += 1
            continue
        if c in _VOWEL:
            out.append(_VOWEL[c]); i += 1; continue
        if c in _MATRA:
            out.append(_MATRA[c]); i += 1; continue
        if c in _SIGN:
            out.append(_SIGN[c]); i += 1; continue
        if c in _DIGIT:
            out.append(_DIGIT[c]); i += 1; continue
        if c == _VIRAMA:
            i += 1; continue
        out.append(c); i += 1
    return "".join(out)


def transliterate(s: str) -> str:
    """Transliterate any Indic runs inside `s`, leaving other text untouched."""
    if not s or not _HAS_INDIC.search(s):
        return s
    s = _fold(s)
    return "".join(
        translit_word(tok) if _HAS_INDIC.search(tok) else tok
        for tok in re.split(r"(\s+)", s)
    )
