"""Pairwise features for (Source-1 record, Source-2/3 candidate)."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def _rowdot(X, i, j, bs=200_000):
    out = np.empty(len(i), np.float32)
    for s in range(0, len(i), bs):
        a, b = X[i[s:s + bs]], X[j[s:s + bs]]
        out[s:s + bs] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def _pair_scores(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, dtype=np.float32, workers=-1)


def _jaccard(B, i, j):
    inter = _rowdot(B, i, j)
    sz = np.asarray(B.sum(axis=1)).ravel().astype(np.float32)
    union = sz[i] + sz[j] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(union > 0, inter / union, np.nan)
    return r.astype(np.float32)


def build_features(rec, pairs):
    """rec: all records with normalised columns. pairs: DataFrame[s1, cand] (global idx)."""
    i = pairs["s1"].to_numpy()
    j = pairs["cand"].to_numpy()
    F = pd.DataFrame(index=pairs.index)

    def tf(analyzer, col, **kw):
        v = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True, dtype=np.float32,
                            token_pattern=r"\S+" if analyzer == "word" else None, **kw)
        return v.fit_transform(rec[col])

    allt = rec["name_n"] + " " + rec["addr_n"]
    F["name_char_cos"] = _rowdot(tf("char_wb", "name_core", ngram_range=(2, 4)), i, j)
    F["name_full_char_cos"] = _rowdot(tf("char_wb", "name_n", ngram_range=(3, 5)), i, j)
    F["name_word_cos"] = _rowdot(tf("word", "name_core"), i, j)
    F["addr_char_cos"] = _rowdot(tf("char_wb", "addr_n", ngram_range=(3, 4)), i, j)
    F["addr_word_cos"] = _rowdot(tf("word", "addr_n"), i, j)
    v = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True, dtype=np.float32)
    F["all_word_cos"] = _rowdot(v.fit_transform(allt), i, j)

    for col, name in (("name_core", "name_jac"), ("addr_n", "addr_jac")):
        B = CountVectorizer(analyzer="word", token_pattern=r"\S+", binary=True,
                            dtype=np.float32).fit_transform(rec[col])
        F[name] = _jaccard(B, i, j)
    Bnum = CountVectorizer(analyzer="word", token_pattern=r"\b\d+\b", binary=True,
                           dtype=np.float32).fit_transform(rec["addr_n"])
    F["addr_num_jac"] = _jaccard(Bnum, i, j)

    # string metrics
    def col(c, idx):
        return rec[c].to_numpy()[idx].tolist()
    nc1, nc2 = col("name_core", i), col("name_core", j)
    nf1, nf2 = col("name_n", i), col("name_n", j)
    ad1, ad2 = col("addr_n", i), col("addr_n", j)
    for nm, sc in (("ratio", fuzz.ratio), ("tsort", fuzz.token_sort_ratio),
                   ("tset", fuzz.token_set_ratio), ("partial", fuzz.partial_ratio),
                   ("jw", JaroWinkler.similarity), ("lev", Levenshtein.normalized_similarity)):
        F[f"name_{nm}"] = _pair_scores(nc1, nc2, sc)
    F["name_full_ratio"] = _pair_scores(nf1, nf2, fuzz.ratio)
    F["name_full_tset"] = _pair_scores(nf1, nf2, fuzz.token_set_ratio)
    a_missing = np.array([(not a) or (not b) for a, b in zip(ad1, ad2)])
    for nm, sc in (("ratio", fuzz.ratio), ("tsort", fuzz.token_sort_ratio),
                   ("tset", fuzz.token_set_ratio), ("jw", JaroWinkler.similarity)):
        s = _pair_scores(ad1, ad2, sc)
        s[a_missing] = np.nan
        F[f"addr_{nm}"] = s
    for c in ("addr_char_cos", "addr_word_cos", "addr_jac", "addr_num_jac"):
        F.loc[a_missing, c] = np.nan

    # exact / structural flags
    def eq(c, empty_nan=True):
        a, b = rec[c].to_numpy()[i], rec[c].to_numpy()[j]
        r = (a == b).astype(np.float32)
        if empty_nan:
            r[(a == "") | (b == "")] = np.nan
        return r
    F["core_eq"] = eq("name_core")
    F["name_eq"] = eq("name_n")
    F["sorted_eq"] = eq("name_sorted")
    F["initials_eq"] = eq("initials")
    F["first_tok_eq"] = eq("first_tok")
    F["last_tok_eq"] = eq("last_tok")
    F["legal_eq"] = eq("legal")
    F["postcode_eq"] = eq("postcode")
    F["house_eq"] = eq("house_no")
    F["same_country"] = (rec["country_n"].to_numpy()[i] == rec["country_n"].to_numpy()[j]).astype(np.float32)
    F["cand_is_s3"] = (rec["src"].to_numpy()[j] == "S3").astype(np.float32)

    # lengths / missingness
    ln = rec["name_core"].str.len().to_numpy()
    la = rec["addr_n"].str.len().to_numpy()
    F["len_name_s1"], F["len_name_c"] = ln[i], ln[j]
    F["len_name_diff"] = np.abs(ln[i] - ln[j])
    F["len_addr_s1"], F["len_addr_c"] = la[i], la[j]
    F["has_pc_s1"] = (rec["postcode"].to_numpy()[i] != "").astype(np.float32)
    F["has_pc_c"] = (rec["postcode"].to_numpy()[j] != "").astype(np.float32)

    # competition among candidates (both directions) -- key for precision
    F["s1"], F["cand"] = i, j
    F["comb"] = (F["name_char_cos"] + F["all_word_cos"]) / 2
    g = F.groupby("s1")["comb"]
    F["rank_s1"] = g.rank(ascending=False, method="min")
    F["gap_s1"] = F["comb"] - g.transform("max")
    h = F.groupby("cand")["comb"]
    F["rank_cand"] = h.rank(ascending=False, method="min")
    F["gap_cand"] = F["comb"] - h.transform("max")
    return F.drop(columns=["s1", "cand"]).astype(np.float32)
