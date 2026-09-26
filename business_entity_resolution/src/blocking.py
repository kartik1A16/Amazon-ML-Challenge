"""Candidate generation (blocking). Union of three channels per Source-1 entity:
  A. char n-gram TF-IDF top-K on the core business name
  B. word TF-IDF top-K on name + address
  C. same postcode (5/6-digit token), ranked by name similarity
Blocking is done inside a country group when `by_country` is True.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def _topk(Xq, Xd, K, chunk=800):
    """Top-K cosine neighbours of each row of Xq among rows of Xd (both L2-normalised, sparse).
    Returns local (q, d) index arrays."""
    if Xq.shape[0] == 0 or Xd.shape[0] == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    XdT = Xd.T.tocsr()
    qs, ds = [], []
    for s in range(0, Xq.shape[0], chunk):
        S = (Xq[s:s + chunk] @ XdT).tocsr()
        for r in range(S.shape[0]):
            a, b = S.indptr[r], S.indptr[r + 1]
            if a == b:
                continue
            d, ix = S.data[a:b], S.indices[a:b]
            if len(d) > K:
                sel = np.argpartition(-d, K - 1)[:K]
                ix = ix[sel]
            qs.append(np.full(len(ix), s + r, np.int64))
            ds.append(ix.astype(np.int64))
    if not qs:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(qs), np.concatenate(ds)


def generate_candidates(rec, k_name=30, k_word=30, k_pc=10, pc_cap=5000,
                        by_country=True, verbose=True):
    """rec: DataFrame of all records (S1+S2+S3) with normalised columns and a RangeIndex.
    Returns DataFrame[s1, cand] of global row indices."""
    n = len(rec)
    max_df = max(int(0.05 * n), 200)  # prune ubiquitous n-grams so sparse products stay cheap
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True,
                         max_df=max_df, dtype=np.float32)
    Xn = vc.fit_transform(rec["name_core"])
    vw = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True,
                         max_df=max_df, dtype=np.float32)
    Xw = vw.fit_transform(rec["name_n"] + " " + rec["addr_n"])

    is_s1 = (rec["src"] == "S1").to_numpy()
    groups = rec["country_n"].to_numpy() if by_country else np.zeros(n, dtype=object)
    out_s, out_c = [], []
    for g in pd.unique(groups):
        gm = groups == g
        q = np.flatnonzero(gm & is_s1)
        d = np.flatnonzero(gm & ~is_s1)
        if len(q) == 0 or len(d) == 0:
            continue
        for X, K in ((Xn, k_name), (Xw, k_word)):
            qi, di = _topk(X[q], X[d], K)
            out_s.append(q[qi]); out_c.append(d[di])
        # channel C: postcode
        pc = rec["postcode"].to_numpy()
        by_pc = {}
        for j in d:
            if pc[j]:
                by_pc.setdefault(pc[j], []).append(j)
        for i in q:
            if not pc[i] or pc[i] not in by_pc:
                continue
            grp = np.asarray(by_pc[pc[i]])
            if len(grp) > pc_cap:
                continue
            if len(grp) > k_pc:
                sims = (Xn[i] @ Xn[grp].T).toarray().ravel()
                grp = grp[np.argpartition(-sims, k_pc - 1)[:k_pc]]
            out_s.append(np.full(len(grp), i)); out_c.append(grp)

    if not out_s:
        return pd.DataFrame({"s1": np.empty(0, np.int64), "cand": np.empty(0, np.int64)})
    s = np.concatenate(out_s).astype(np.int64)
    c = np.concatenate(out_c).astype(np.int64)
    key = np.unique(s * n + c)
    pairs = pd.DataFrame({"s1": key // n, "cand": key % n})
    if verbose:
        n1 = int(is_s1.sum())
        print(f"[blocking] {len(pairs):,} candidate pairs for {n1:,} S1 entities "
              f"({len(pairs) / max(n1, 1):.1f} per entity)")
    return pairs
