"""Language-agnostic text normalisation for business names and addresses.

No external lookups: only unicode tables and hand-written abbreviation lists.
`anyascii` (ISC licence, fully offline) is used when installed to transliterate
non-Latin scripts and strip accents; otherwise we fall back to NFKD accent stripping.
"""
import re
import unicodedata

try:  # optional, offline transliteration table
    from anyascii import anyascii as _anyascii
except Exception:  # pragma: no cover
    _anyascii = None

# Legal-form tokens (after NAME_MAP canonicalisation). Split off from the "core" name.
LEGAL = {
    "incorporated", "corporation", "limited", "private", "company",
    "llc", "llp", "lp", "pllc", "plc", "opc", "ltda",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scop",
    "gmbh", "ag", "kg", "ug", "bv", "nv", "srl", "spa", "sl", "ab", "oy",
    "pty", "proprietary", "holdings", "group", "enterprises", "enterprise",
}
# Function words dropped from the core name only.
STOP = {"and", "et", "und", "y", "the", "le", "la", "les", "l", "de", "du", "des", "d", "of"}

NAME_MAP = {
    "corp": "corporation", "inc": "incorporated", "ltd": "limited", "lt": "limited",
    "pvt": "private", "co": "company", "comp": "company", "intl": "international",
    "assoc": "associates", "svcs": "services", "svc": "services", "mfg": "manufacturing",
    "ent": "enterprises", "ents": "enterprises", "bros": "brothers",
    "shri": "shree", "sri": "shree", "laxmi": "lakshmi", "lakshmi": "lakshmi",
    "sons": "son", "&": "and",
}

ADDR_MAP = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bvd": "boulevard", "bd": "boulevard", "dr": "drive",
    "ln": "lane", "ct": "court", "pl": "place", "sq": "square", "hwy": "highway",
    "pkwy": "parkway", "ste": "suite", "apt": "apartment", "fl": "floor",
    "flr": "floor", "bldg": "building", "nr": "near", "opp": "opposite",
    "bhd": "behind", "sec": "sector", "ph": "phase", "dist": "district",
    "mkt": "market", "indl": "industrial", "chem": "chemin", "imp": "impasse",
    "fbg": "faubourg", "no": "number", "nagar": "nagar", "ngr": "nagar",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
}

_DOTTED = re.compile(r"\b(?:[^\W\d_]\.){2,}")
_PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)
_WS = re.compile(r"\s+")


def norm_text(s: str) -> str:
    """Unicode-normalise, transliterate/strip accents, casefold, drop punctuation."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    if _anyascii is not None:
        s = _anyascii(s)
    else:
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = s.casefold()
    s = _DOTTED.sub(lambda m: m.group().replace(".", ""), s)  # "p.v.t." -> "pvt"
    s = s.replace("&", " and ")
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def norm_name(s: str):
    """Return (full_name, core_name, legal_suffixes)."""
    t = norm_text(s)
    t = re.sub(r"\bp ltd\b", "private limited", t)
    toks = [NAME_MAP.get(w, w) for w in t.split()]
    full = " ".join(toks)
    core = [w for w in toks if w not in LEGAL and w not in STOP]
    legal = sorted({w for w in toks if w in LEGAL})
    core_s = " ".join(core) if core else full
    return full, core_s, " ".join(legal)


_PC = re.compile(r"\d{5,6}")
_HN = re.compile(r"\d{1,4}[a-z]?")


def norm_addr(s: str):
    """Return (address_norm, postcode, house_number). Postcode = last 5/6-digit token
    (covers US ZIP, India PIN, French code postal without naming any country)."""
    t = norm_text(s)
    toks = [ADDR_MAP.get(w, w) for w in t.split()]
    pcs = [w for w in toks if _PC.fullmatch(w)]
    pc = pcs[-1] if pcs else ""
    hn = next((w for w in toks if _HN.fullmatch(w) and w != pc), "")
    return " ".join(toks), pc, hn


def add_norm_columns(df):
    """df needs: entity_id, business_name, business_address, country."""
    names = [norm_name(x) for x in df["business_name"].tolist()]
    addrs = [norm_addr(x) for x in df["business_address"].tolist()]
    df["name_n"] = [n[0] for n in names]
    df["name_core"] = [n[1] for n in names]
    df["legal"] = [n[2] for n in names]
    df["addr_n"] = [a[0] for a in addrs]
    df["postcode"] = [a[1] for a in addrs]
    df["house_no"] = [a[2] for a in addrs]
    df["country_n"] = [norm_text(c) for c in df["country"].tolist()]
    core_tok = [c.split() for c in df["name_core"].tolist()]
    df["name_sorted"] = [" ".join(sorted(t)) for t in core_tok]
    df["initials"] = ["".join(w[0] for w in t) for t in core_tok]
    df["first_tok"] = [t[0] if t else "" for t in core_tok]
    df["last_tok"] = [t[-1] if t else "" for t in core_tok]
    df["src"] = df["entity_id"].str[:2]
    return df
