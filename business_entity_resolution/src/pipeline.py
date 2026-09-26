"""End-to-end baseline: data -> normalisation -> blocking -> pairwise LightGBM -> F0.5-tuned
threshold -> output/matching_results.tsv + output/candidate_pairs.tsv

Usage (from the code/business_entity_resolution directory):
    python src/pipeline.py --data-dir ../../dataset --out-dir ../../output --work-dir work
"""
import argparse
import csv
import os
import pickle
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import generate_candidates  # noqa: E402
from features import build_features  # noqa: E402
from metrics import macro_f05  # noqa: E402
from normalize import add_norm_columns  # noqa: E402

warnings.filterwarnings("ignore", message=".*eval_set.*")

LGB_PARAMS = dict(
    learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1,
)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- IO
def read_tsv(path):
    # explicit tab separator; no quote handling (addresses may contain quotes/commas)
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
                       encoding="utf-8")


def load_records(folder, prefix):
    frames = []
    for k in (1, 2, 3):
        df = read_tsv(os.path.join(folder, f"{prefix}_source{k}.tsv"))
        for c in ("entity_id", "business_name", "business_address", "country"):
            if c not in df.columns:
                raise ValueError(f"{prefix}_source{k}.tsv is missing column {c}; got {list(df.columns)}")
        df["entity_id"] = df["entity_id"].str.strip()
        frames.append(df[["entity_id", "business_name", "business_address", "country"]])
    rec = pd.concat(frames, ignore_index=True)
    return add_norm_columns(rec)


def load_truth(path):
    gt = read_tsv(path)
    truth = {}
    for s, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        truth[s.strip()] = {x.strip() for x in m.split(",") if x.strip()}
    return truth


def write_tsv(path, s1_ids, mapping, col2):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col2}\n")
        for s in s1_ids:
            f.write(s + "\t" + ",".join(sorted(mapping.get(s, ()))) + "\n")


# ------------------------------------------------------------------------ evaluation
def per_entity_f05(n1, s1p, y, p_eff, ntrue, thr):
    sel = p_eff >= thr
    npred = np.bincount(s1p[sel], minlength=n1).astype(float)
    tp = np.bincount(s1p[sel & y], minlength=n1).astype(float)
    P = np.divide(tp, npred, out=np.zeros(n1), where=npred > 0)
    R = np.divide(tp, ntrue, out=np.zeros(n1), where=ntrue > 0)
    den = 0.25 * P + R
    F = np.divide(1.25 * P * R, den, out=np.zeros(n1), where=den > 0)
    return np.where(ntrue == 0, (npred == 0).astype(float), F)


def resolve_scores(pairs, p):
    """Each S2/S3 record may belong to only one (deduplicated) S1 entity: keep the best."""
    best = pd.Series(p).groupby(pairs["cand"].to_numpy()).transform("max").to_numpy()
    return np.where(p >= best, p, -1.0)


def tune_threshold(n1, s1p, y, p, pairs, ntrue, grid=np.arange(0.10, 0.981, 0.01)):
    best = (-1, None, None)
    res = {}
    for resolve in (True, False):
        pe = resolve_scores(pairs, p) if resolve else p
        for t in grid:
            f = per_entity_f05(n1, s1p, y, pe, ntrue, t).mean()
            res[(resolve, round(float(t), 2))] = f
            if f > best[0]:
                best = (f, resolve, float(t))
    return best, res


def fit_lgb(X, y, Xv=None, yv=None, n_estimators=2000, seed=0):
    m = lgb.LGBMClassifier(n_estimators=n_estimators, random_state=seed, **LGB_PARAMS)
    if Xv is not None:
        m.fit(X, y, eval_set=[(Xv, yv)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        m.fit(X, y)
    return m


# ------------------------------------------------------------------------------ main
def build_pairs(rec, args, by_country, cache_path=None):
    if cache_path and args.cache and os.path.exists(cache_path):
        log("loading cached pairs/features", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    pairs = generate_candidates(rec, args.k_name, args.k_word, args.k_pc, by_country=by_country)
    log("building features ...")
    F = build_features(rec, pairs)
    if cache_path and args.cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump((pairs, F), f)
    return pairs, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset", help="folder containing train/ and test/")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--work-dir", default="work", help="caches / diagnostics (not part of submission)")
    ap.add_argument("--k-name", type=int, default=30)
    ap.add_argument("--k-word", type=int, default=30)
    ap.add_argument("--k-pc", type=int, default=10)
    ap.add_argument("--by-country", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--thr", type=float, default=None, help="override tuned probability threshold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", action="store_true")
    args = ap.parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)

    # ------------------------------------------------------------ train
    log("loading train")
    tr = load_records(os.path.join(args.data_dir, "train"), "train")
    truth = load_truth(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))
    ids = tr["entity_id"].to_numpy()
    idset = set(ids)
    s1_rows = np.flatnonzero((tr["src"] == "S1").to_numpy())
    n1 = len(s1_rows)
    s1_pos = np.full(len(tr), -1, np.int64)
    s1_pos[s1_rows] = np.arange(n1)
    truth = {k: {x for x in v if x in idset} for k, v in truth.items() if k in idset}
    ntrue = np.array([len(truth.get(ids[r], ())) for r in s1_rows], dtype=float)

    # ground-truth diagnostics
    owners = {}
    for s, ms in truth.items():
        for m in ms:
            owners.setdefault(m, set()).add(s)
    cn = dict(zip(tr["entity_id"], tr["country_n"]))
    tot = int(ntrue.sum())
    same_c = sum(cn[s] == cn[m] for s, ms in truth.items() for m in ms)
    log(f"train: {n1:,} S1 | singletons {np.mean(ntrue == 0):.1%} | matches/entity {ntrue.mean():.2f} | "
        f"S2 {sum(m.startswith('S2') for ms in truth.values() for m in ms):,} "
        f"S3 {sum(m.startswith('S3') for ms in truth.values() for m in ms):,} | "
        f"cands owned by >1 S1: {sum(len(v) > 1 for v in owners.values())} | "
        f"same-country true pairs {same_c / max(tot, 1):.2%} | countries {sorted(set(tr['country_n']))}")
    by_country = {"yes": True, "no": False}.get(args.by_country, same_c / max(tot, 1) >= 0.99)
    log("blocking within country:", by_country)

    pairs, F = build_pairs(tr, args, by_country, os.path.join(args.work_dir, "train_pairs.pkl"))
    y = np.array([ids[c] in truth.get(ids[s], ()) for s, c in zip(pairs["s1"], pairs["cand"])])
    s1p = s1_pos[pairs["s1"].to_numpy()]
    log(f"blocking recall on train: {y.sum() / max(tot, 1):.4f} "
        f"({int(y.sum()):,}/{tot:,}); positives in pairs {y.mean():.3%}")

    # ------------------------------------------------ out-of-fold predictions (grouped by S1)
    oof = np.zeros(len(F))
    best_iters = []
    for k, (a, b) in enumerate(GroupKFold(args.folds).split(F, y, groups=s1p)):
        m = fit_lgb(F.iloc[a], y[a], F.iloc[b], y[b], seed=args.seed)
        oof[b] = m.predict_proba(F.iloc[b])[:, 1]
        best_iters.append(m.best_iteration_ or m.n_estimators)
        log(f"fold {k}: best_iter {best_iters[-1]}")
    (f_best, resolve, thr), grid = tune_threshold(n1, s1p, y, oof, pairs, ntrue)
    log(f"OOF macro F0.5 = {f_best:.4f}  (resolve={resolve}, thr={thr:.2f}); "
        f"no-resolve best = {max(v for (r, _), v in grid.items() if not r):.4f}")
    if args.thr is not None:
        thr = args.thr
    pe = resolve_scores(pairs, oof) if resolve else oof
    fs = per_entity_f05(n1, s1p, y, pe, ntrue, thr)
    ctry = tr["country_n"].to_numpy()[s1_rows]
    for c in sorted(set(ctry)):
        log(f"  OOF F0.5 [{c}] = {fs[ctry == c].mean():.4f}")
    # sanity check against the set-based reference scorer
    sel = pe >= thr
    pred = {}
    for s, c in zip(pairs["s1"].to_numpy()[sel], pairs["cand"].to_numpy()[sel]):
        pred.setdefault(ids[s], set()).add(ids[c])
    ref = macro_f05([ids[r] for r in s1_rows], pred, truth)
    assert abs(ref - fs.mean()) < 1e-9, (ref, fs.mean())

    # ------------------------------------------- leave-one-country-out robustness (France proxy)
    countries = sorted(set(ctry))
    if len(countries) > 1:
        s1_ctry_of_pair = pd.Series(tr["country_n"].to_numpy()[pairs["s1"].to_numpy()])
        for hold in countries:
            te_m = (s1_ctry_of_pair == hold).to_numpy()
            if te_m.all() or not te_m.any():
                continue
            m = fit_lgb(F[~te_m], y[~te_m], F[te_m], y[te_m], seed=args.seed)
            p_h = np.zeros(len(F)); p_h[te_m] = m.predict_proba(F[te_m])[:, 1]
            pe_h = resolve_scores(pairs, np.where(te_m, p_h, 0.0)) if resolve else p_h
            f_h = per_entity_f05(n1, s1p, y, pe_h, ntrue, thr)[ctry == hold].mean()
            log(f"  LOCO train on others -> test on [{hold}] at thr {thr:.2f}: F0.5 = {f_h:.4f}")

    # ------------------------------------------------------------------ final model
    n_est = max(int(np.mean(best_iters) * 1.1), 50)
    final = fit_lgb(F, y, n_estimators=n_est, seed=args.seed)
    imp = pd.Series(final.booster_.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
    imp.to_csv(os.path.join(args.work_dir, "feature_importance.tsv"), sep="\t")
    final.booster_.save_model(os.path.join(args.work_dir, "model.txt"))

    # ------------------------------------------------------------------ test
    log("loading test")
    te = load_records(os.path.join(args.data_dir, "test"), "test")
    tids = te["entity_id"].to_numpy()
    te_s1 = np.flatnonzero((te["src"] == "S1").to_numpy())
    s1_ids = [tids[r] for r in te_s1]
    if len(set(s1_ids)) != len(s1_ids):
        raise ValueError("duplicate Source 1 ids in test")
    log(f"test: {len(te_s1):,} S1 | countries {sorted(set(te['country_n']))}")
    tpairs, TF = build_pairs(te, args, by_country, os.path.join(args.work_dir, "test_pairs.pkl"))
    p = final.predict_proba(TF[F.columns])[:, 1]
    pe_t = resolve_scores(tpairs, p) if resolve else p
    sel = pe_t >= thr
    matches, cands = {}, {}
    for s, c in zip(tpairs["s1"].to_numpy(), tpairs["cand"].to_numpy()):
        cands.setdefault(tids[s], set()).add(tids[c])
    for s, c in zip(tpairs["s1"].to_numpy()[sel], tpairs["cand"].to_numpy()[sel]):
        matches.setdefault(tids[s], set()).add(tids[c])
    write_tsv(os.path.join(args.out_dir, "matching_results.tsv"), s1_ids, matches, "matched_entity_ids")
    write_tsv(os.path.join(args.out_dir, "candidate_pairs.tsv"), s1_ids, cands, "candidate_entity_ids")

    # shift diagnostics: predicted match rate per country vs. train singleton rate
    tc = te["country_n"].to_numpy()
    for c in sorted(set(tc[te_s1])):
        rows = [tids[r] for r in te_s1 if tc[r] == c]
        rate = np.mean([bool(matches.get(s)) for s in rows])
        log(f"  test [{c}]: {len(rows):,} S1, predicted non-singleton {rate:.1%}, "
            f"avg matches {np.mean([len(matches.get(s, ())) for s in rows]):.2f}")
    log(f"wrote {args.out_dir}/matching_results.tsv and candidate_pairs.tsv "
        f"({sum(bool(v) for v in matches.values()):,} S1 with matches)")


if __name__ == "__main__":
    main()
