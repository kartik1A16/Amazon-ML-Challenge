#!/usr/bin/env python3
"""
End-to-end entity resolution pipeline for the Business Entity Resolution Challenge.

This pipeline:
1. Reads train/test TSV files.
2. Cleans and normalizes business names, addresses, and countries.
3. Generates blocking candidates for every Source 1 entity.
4. Computes pairwise similarity features.
5. Trains a LightGBM matching model.
6. Tunes a decision threshold using macro F_0.5 on a validation split.
7. Writes matching_results.tsv and candidate_pairs.tsv.

No synthetic data is generated.
No external APIs or external business lookup services are used.
"""

import argparse
import csv
import logging
import math
import re
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from rapidfuzz import fuzz

try:
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("entity_resolution")

# ------------------------------------------------------------------------------
# Global vectorizer for blocking.
# HashingVectorizer is fit-free, deterministic, and avoids train/test vocabulary
# leakage. Character n-grams are robust to typos and partial string variations.
# ------------------------------------------------------------------------------
VECTOR = HashingVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 4),
    n_features=2**17,
    alternate_sign=True,
    norm="l2",
    include_bias=False,
)

# Conservative offline normalization rules.
ABBREVIATIONS = {
    r"\bpvt\b": "private",
    r"\bpriv\b": "private",
    r"\bltd\b": "limited",
    r"\blim\b": "limited",
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bco\b": "company",
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
}
ABBREV_PATTERNS = [(re.compile(k), v) for k, v in ABBREVIATIONS.items()]

FEATURE_COLUMNS = [
    "retrieval_score",
    "country_match",
    "country_mismatch",
    "country_both_missing",
    "cand_is_s2",
    "cand_is_s3",
    "name_token_set",
    "name_token_sort",
    "name_partial",
    "name_wratio",
    "addr_token_set",
    "addr_token_sort",
    "addr_partial",
    "addr_wratio",
    "name_jaccard_tokens",
    "addr_jaccard_tokens",
    "name_jaccard_char3",
    "addr_jaccard_char3",
    "name_len_diff",
    "name_len_ratio",
    "addr_len_diff",
    "addr_len_ratio",
    "num_jaccard",
    "num_overlap",
    "num_count_sum",
    "has_num1",
    "has_num2",
    "combined_max",
    "combined_weighted",
    "name_addr_min",
    "both_name_addr_high",
]


# ------------------------------------------------------------------------------
# Basic text cleaning
# ------------------------------------------------------------------------------
def clean_text(value: str) -> str:
    """
    Offline text normalization only.
    No external APIs, no translation services, no geocoding.
    """
    text = str(value or "").strip().lower()
    if not text:
        return ""

    text = text.replace("&", " and ")

    # Remove accents where possible.
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # Keep alphanumeric characters and whitespace.
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip()

    # Normalize common abbreviations.
    for pattern, replacement in ABBREV_PATTERNS:
        text = pattern.sub(replacement, text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def valid_token(token: str) -> bool:
    if not token:
        return False
    if token.isdigit():
        return len(token) >= 2
    return len(token) >= 3


def char_ngrams(text: str, n: int = 3) -> set:
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard_sets(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / float(union)


def fuzzy_score(func, a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    try:
        return float(func(a, b)) / 100.0
    except Exception:
        return 0.0


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------
def read_source(path: Path, expected_prefix: str) -> pd.DataFrame:
    LOGGER.info("Reading %s", path)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    required = ["entity_id", "business_name", "business_address", "country"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    for col in required:
        df[col] = df[col].astype(str).str.strip()

    invalid = ~df["entity_id"].str.startswith(expected_prefix)
    if invalid.any():
        bad_examples = df.loc[invalid, "entity_id"].head(5).tolist()
        raise ValueError(
            f"Invalid entity_id prefix in {path}. Expected prefix {expected_prefix}. "
            f"Examples: {bad_examples}"
        )

    if df["entity_id"].duplicated().any():
        raise ValueError(f"Duplicate entity_id values found in {path}")

    return df


def read_ground_truth(path: Path, all_source1_ids: list) -> dict:
    LOGGER.info("Reading ground truth %s", path)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    required = ["source1_entity_id", "matched_entity_ids"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required ground truth columns: {missing}")

    df["source1_entity_id"] = df["source1_entity_id"].astype(str).str.strip()
    df["matched_entity_ids"] = df["matched_entity_ids"].astype(str).str.strip()

    if df["source1_entity_id"].duplicated().any():
        raise ValueError("Duplicate source1_entity_id values found in ground truth")

    gt_map = {sid: set() for sid in all_source1_ids}

    for sid, ids in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        if ids:
            gt_map[sid] = {x.strip() for x in ids.split(",") if x.strip()}
        else:
            gt_map[sid] = set()

    return gt_map


# ------------------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------------------
def preprocess_source_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds cleaned fields, tokens, numbers, and vector text.
    All operations are local and deterministic.
    """
    df = df.copy()

    required = ["entity_id", "business_name", "business_address", "country"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        df[col] = df[col].astype(str).str.strip()

    if df["entity_id"].duplicated().any():
        raise ValueError("Duplicate entity_id found during preprocessing")

    names = [clean_text(x) for x in df["business_name"]]
    addresses = [clean_text(x) for x in df["business_address"]]
    countries = [clean_text(x) for x in df["country"]]

    name_tokens = [x.split() if x else [] for x in names]
    address_tokens = [x.split() if x else [] for x in addresses]
    address_numbers = [re.findall(r"\d+", x) for x in addresses]

    all_tokens = []
    query_tokens = []

    for nt, at in zip(name_tokens, address_tokens):
        seen = set()
        combined = []
        for tok in nt + at:
            if valid_token(tok) and tok not in seen:
                seen.add(tok)
                combined.append(tok)
        all_tokens.append(combined)

        qseen = set()
        qt = []
        for tok in nt[:12] + at[:18]:
            if valid_token(tok) and tok not in qseen:
                qseen.add(tok)
                qt.append(tok)
        query_tokens.append(qt)

    text_for_vector = [
        f"{name} {name} {addr} {country}".strip()
        for name, addr, country in zip(names, addresses, countries)
    ]

    df["business_name_clean"] = names
    df["business_address_clean"] = addresses
    df["country_clean"] = countries
    df["name_tokens"] = name_tokens
    df["address_tokens"] = address_tokens
    df["business_address_numbers"] = address_numbers
    df["all_tokens"] = all_tokens
    df["query_tokens"] = query_tokens
    df["text_for_vector"] = text_for_vector

    return df


def make_record_map(df: pd.DataFrame) -> dict:
    if df["entity_id"].duplicated().any():
        raise ValueError("Duplicate entity_id found while creating record map")
    return df.set_index("entity_id").to_dict(orient="index")


# ------------------------------------------------------------------------------
# Blocking / candidate generation
# ------------------------------------------------------------------------------
def build_exact_maps(pool_df: pd.DataFrame):
    name_map = defaultdict(list)
    addr_map = defaultdict(list)

    for eid, name, addr in zip(
        pool_df["entity_id"],
        pool_df["business_name_clean"],
        pool_df["business_address_clean"],
    ):
        if name:
            name_map[name].append(eid)
        if addr:
            addr_map[addr].append(eid)

    return name_map, addr_map


def build_token_map(pool_df: pd.DataFrame, token_df_limit: int) -> dict:
    """
    Builds an inverted token index, keeping only tokens whose document frequency
    is below token_df_limit. This avoids extremely common tokens.
    """
    token_counts = Counter()
    for tokens in pool_df["all_tokens"]:
        if tokens:
            token_counts.update(set(tokens))

    allowed_tokens = {
        tok for tok, cnt in token_counts.items() if cnt <= token_df_limit
    }

    if not allowed_tokens:
        return {}

    token_map = defaultdict(list)
    for eid, tokens in zip(pool_df["entity_id"], pool_df["all_tokens"]):
        if not tokens:
            continue
        for tok in set(tokens):
            if tok in allowed_tokens:
                token_map[tok].append(eid)

    return token_map


def add_token_candidates(
    cand_dict: dict,
    query_tokens_list: list,
    token_map: dict,
    top_token_candidates: int,
    ids_per_token: int,
) -> None:
    scores = defaultdict(float)

    for tok in query_tokens_list:
        ids = token_map.get(tok)
        if not ids:
            continue

        weight = 1.0 / math.log(1.0 + float(len(ids)))
        for cid in ids[:ids_per_token]:
            scores[cid] += weight

    if not scores:
        return

    top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[
        :top_token_candidates
    ]
    max_score = max(score for _, score in top)
    if max_score <= 0:
        return

    for cid, score in top:
        normalized = 0.95 * (score / max_score)
        if normalized > cand_dict.get(cid, -1.0):
            cand_dict[cid] = normalized


def generate_candidates(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    """
    Generates final blocking candidates for every Source 1 entity.

    The returned dictionary is:
        {
            source1_entity_id: {
                candidate_entity_id: blocking_score,
                ...
            },
            ...
        }

    This is the exact candidate set that will be fed into the model.
    """
    s1_ids = s1_df["entity_id"].tolist()
    candidates = {sid: {} for sid in s1_ids}

    if len(s1_df) == 0 or len(pool_df) == 0:
        return candidates

    LOGGER.info("Vectorizing Source 1 and pool records for blocking")
    pool_mat = VECTOR.transform(pool_df["text_for_vector"].tolist())
    s1_mat = VECTOR.transform(s1_df["text_for_vector"].tolist())

    top_k = min(args.top_k, len(pool_df))

    if top_k > 0:
        LOGGER.info("Running nearest-neighbor blocking with top_k=%d", top_k)
        nn = NearestNeighbors(
            n_neighbors=top_k,
            metric="cosine",
            algorithm="brute",
            n_jobs=args.nn_jobs,
        )
        nn.fit(pool_mat)

        distances, indices = nn.kneighbors(s1_mat)
        scores = np.nan_to_num(1.0 - distances, nan=0.0, posinf=0.0, neginf=0.0)

        pool_ids = pool_df["entity_id"].tolist()

        for i, sid in enumerate(s1_ids):
            cand_dict = candidates[sid]
            for j, idx in enumerate(indices[i]):
                idx = int(idx)
                if idx < 0 or idx >= len(pool_ids):
                    continue

                score = float(scores[i][j])
                if score < 0.0:
                    score = 0.0
                if score > 1.0:
                    score = 1.0

                cid = pool_ids[idx]
                if score > cand_dict.get(cid, -1.0):
                    cand_dict[cid] = score

    LOGGER.info("Building exact-match blocking maps")
    name_map, addr_map = build_exact_maps(pool_df)

    s1_names = s1_df["business_name_clean"].tolist()
    s1_addrs = s1_df["business_address_clean"].tolist()

    for sid, name, addr in zip(s1_ids, s1_names, s1_addrs):
        cand_dict = candidates[sid]

        if name:
            exact_ids = name_map.get(name, [])
            if 0 < len(exact_ids) <= args.exact_key_limit:
                for cid in exact_ids[: args.exact_per_key]:
                    if 1.0 > cand_dict.get(cid, -1.0):
                        cand_dict[cid] = 1.0

        if addr:
            exact_ids = addr_map.get(addr, [])
            if 0 < len(exact_ids) <= args.exact_key_limit:
                for cid in exact_ids[: args.exact_per_key]:
                    if 1.0 > cand_dict.get(cid, -1.0):
                        cand_dict[cid] = 1.0

    if not args.disable_token_blocking:
        LOGGER.info("Building token blocking map")
        token_map = build_token_map(pool_df, args.token_df_limit)
        top_token_candidates = max(10, args.top_k // 2)

        s1_query_tokens = s1_df["query_tokens"].tolist()

        for sid, qt in zip(s1_ids, s1_query_tokens):
            add_token_candidates(
                candidates[sid],
                qt,
                token_map,
                top_token_candidates,
                args.token_ids_per_token,
            )

    # Trim to final candidate set.
    if args.max_candidates > 0:
        for sid in list(candidates.keys()):
            cand_dict = candidates[sid]
            if len(cand_dict) > args.max_candidates:
                top = sorted(
                    cand_dict.items(),
                    key=lambda item: (-item[1], item[0]),
                )[: args.max_candidates]
                candidates[sid] = dict(top)

    return candidates


def log_candidate_stats(candidates: dict, name: str) -> None:
    counts = [len(v) for v in candidates.values()]
    if not counts:
        LOGGER.info("%s: no entities", name)
        return

    LOGGER.info(
        "%s: entities=%d total_candidates=%d avg=%.2f max=%d zero_candidate_entities=%d",
        name,
        len(counts),
        int(sum(counts)),
        float(np.mean(counts)),
        int(max(counts)),
        int(sum(1 for c in counts if c == 0)),
    )


# ------------------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------------------
def extract_features(
    s1_rec: dict,
    cand_rec: dict,
    retrieval_score: float,
    candidate_id: str,
) -> dict:
    name1 = str(s1_rec.get("business_name_clean", ""))
    name2 = str(cand_rec.get("business_name_clean", ""))

    addr1 = str(s1_rec.get("business_address_clean", ""))
    addr2 = str(cand_rec.get("business_address_clean", ""))

    country1 = str(s1_rec.get("country_clean", ""))
    country2 = str(cand_rec.get("country_clean", ""))

    try:
        retrieval_score = float(retrieval_score)
        if not np.isfinite(retrieval_score):
            retrieval_score = 0.0
    except Exception:
        retrieval_score = 0.0

    retrieval_score = max(0.0, min(1.0, retrieval_score))

    feats = {}

    feats["retrieval_score"] = retrieval_score
    feats["country_match"] = 1.0 if country1 and country1 == country2 else 0.0
    feats["country_mismatch"] = (
        1.0 if country1 and country2 and country1 != country2 else 0.0
    )
    feats["country_both_missing"] = 1.0 if not country1 and not country2 else 0.0
    feats["cand_is_s2"] = 1.0 if str(candidate_id).startswith("S2-") else 0.0
    feats["cand_is_s3"] = 1.0 if str(candidate_id).startswith("S3-") else 0.0

    name_token_set = fuzzy_score(fuzz.token_set_ratio, name1, name2)
    name_token_sort = fuzzy_score(fuzz.token_sort_ratio, name1, name2)
    name_partial = fuzzy_score(fuzz.partial_ratio, name1, name2)
    name_wratio = fuzzy_score(fuzz.WRatio, name1, name2)

    addr_token_set = fuzzy_score(fuzz.token_set_ratio, addr1, addr2)
    addr_token_sort = fuzzy_score(fuzz.token_sort_ratio, addr1, addr2)
    addr_partial = fuzzy_score(fuzz.partial_ratio, addr1, addr2)
    addr_wratio = fuzzy_score(fuzz.WRatio, addr1, addr2)

    feats["name_token_set"] = name_token_set
    feats["name_token_sort"] = name_token_sort
    feats["name_partial"] = name_partial
    feats["name_wratio"] = name_wratio

    feats["addr_token_set"] = addr_token_set
    feats["addr_token_sort"] = addr_token_sort
    feats["addr_partial"] = addr_partial
    feats["addr_wratio"] = addr_wratio

    name_tokens1 = set(s1_rec.get("name_tokens", []))
    name_tokens2 = set(cand_rec.get("name_tokens", []))
    addr_tokens1 = set(s1_rec.get("address_tokens", []))
    addr_tokens2 = set(cand_rec.get("address_tokens", []))

    feats["name_jaccard_tokens"] = jaccard_sets(name_tokens1, name_tokens2)
    feats["addr_jaccard_tokens"] = jaccard_sets(addr_tokens1, addr_tokens2)

    name_char3_1 = char_ngrams(name1, 3)
    name_char3_2 = char_ngrams(name2, 3)
    addr_char3_1 = char_ngrams(addr1, 3)
    addr_char3_2 = char_ngrams(addr2, 3)

    feats["name_jaccard_char3"] = jaccard_sets(name_char3_1, name_char3_2)
    feats["addr_jaccard_char3"] = jaccard_sets(addr_char3_1, addr_char3_2)

    name_len1 = len(name1)
    name_len2 = len(name2)
    addr_len1 = len(addr1)
    addr_len2 = len(addr2)

    feats["name_len_diff"] = float(abs(name_len1 - name_len2))
    feats["name_len_ratio"] = float(min(name_len1, name_len2) / max(name_len1, name_len2, 1))
    feats["addr_len_diff"] = float(abs(addr_len1 - addr_len2))
    feats["addr_len_ratio"] = float(min(addr_len1, addr_len2) / max(addr_len1, addr_len2, 1))

    num1 = set(s1_rec.get("business_address_numbers", []))
    num2 = set(cand_rec.get("business_address_numbers", []))

    feats["num_jaccard"] = jaccard_sets(num1, num2)
    feats["num_overlap"] = float(len(num1 & num2))
    feats["num_count_sum"] = float(len(num1) + len(num2))
    feats["has_num1"] = 1.0 if num1 else 0.0
    feats["has_num2"] = 1.0 if num2 else 0.0

    name_best = max(name_token_set, name_token_sort, name_partial, name_wratio)
    addr_best = max(addr_token_set, addr_token_sort, addr_partial, addr_wratio)

    feats["combined_max"] = max(name_best, addr_best)
    feats["combined_weighted"] = 0.65 * name_best + 0.35 * addr_best
    feats["name_addr_min"] = min(name_best, addr_best)
    feats["both_name_addr_high"] = (
        1.0 if name_best >= 0.85 and addr_best >= 0.75 else 0.0
    )

    return feats


def build_feature_dataframe(
    entity_ids: list,
    candidates: dict,
    gt_map: dict | None,
    s1_map: dict,
    pool_map: dict,
    add_true_matches: bool = False,
) -> pd.DataFrame:
    rows = []

    for sid in entity_ids:
        s1_rec = s1_map.get(sid)
        if s1_rec is None:
            continue

        true_set = gt_map.get(sid, set()) if gt_map is not None else set()

        cand_dict = dict(candidates.get(sid, {}))

        # Optional training-only augmentation:
        # Include known true positives so the model sees enough positive examples.
        # This does not affect test candidate_pairs.tsv.
        if add_true_matches and gt_map is not None:
            for cid in true_set:
                if cid in pool_map:
                    cand_dict.setdefault(cid, 0.0)

        for cid, score in cand_dict.items():
            cand_rec = pool_map.get(cid)
            if cand_rec is None:
                continue

            feats = extract_features(s1_rec, cand_rec, score, cid)
            feats["source1_entity_id"] = sid
            feats["candidate_entity_id"] = cid

            if gt_map is not None:
                feats["label"] = 1 if cid in true_set else 0

            rows.append(feats)

    if not rows:
        columns = ["source1_entity_id", "candidate_entity_id"]
        if gt_map is not None:
            columns.append("label")
        columns += FEATURE_COLUMNS
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)

    columns = ["source1_entity_id", "candidate_entity_id"]
    if gt_map is not None:
        columns.append("label")
    columns += FEATURE_COLUMNS

    return df[columns]


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
class DummyModel:
    """
    Fallback model that always predicts a fixed positive probability.
    Used when training data has only one class or is empty.
    """

    def __init__(self, positive_probability: float = 0.0):
        self.positive_probability = positive_probability

    def predict_proba(self, X):
        if hasattr(X, "shape"):
            n = X.shape[0]
        else:
            n = len(X)

        pos = np.full(n, self.positive_probability, dtype=float)
        return np.column_stack([1.0 - pos, pos])


def train_model(train_df: pd.DataFrame):
    if train_df.empty or "label" not in train_df.columns:
        LOGGER.warning("Training dataframe empty. Using dummy model.")
        return DummyModel(0.0)

    X = train_df[FEATURE_COLUMNS].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)

    y = train_df["label"].astype(int).to_numpy()

    if len(y) == 0 or len(np.unique(y)) < 2:
        LOGGER.warning("Only one label class in training data. Using dummy model.")
        return DummyModel(0.0)

    pos = int(y.sum())
    neg = len(y) - pos
    n_rows = len(train_df)

    min_child = max(1, min(50, n_rows // 100))

    LOGGER.info(
        "Training model: rows=%d positives=%d negatives=%d",
        n_rows,
        pos,
        neg,
    )

    if HAS_LIGHTGBM:
        scale_pos_weight = min(100.0, max(1.0, neg / max(1, pos)))

        model = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.08,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=min_child,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(X, y)
        return model

    LOGGER.warning("LightGBM not available. Falling back to HistGradientBoosting.")
    sample_weight = np.where(y == 1, min(100.0, neg / max(1, pos)), 1.0)

    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=63,
        min_samples_leaf=min_child,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def predict_proba(model, df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=float)

    X = df[FEATURE_COLUMNS].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)

    proba = model.predict_proba(X)[:, 1]
    proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)
    return proba


# ------------------------------------------------------------------------------
# Evaluation and threshold tuning
# ------------------------------------------------------------------------------
def build_pred_map(df: pd.DataFrame) -> dict:
    pred_map = defaultdict(list)
    if df.empty:
        return pred_map

    for row in df.itertuples(index=False):
        pred_map[row.source1_entity_id].append(
            (row.candidate_entity_id, float(row.prob))
        )

    return pred_map


def prepare_threshold_curves(entity_ids: list, pred_map: dict, gt_map: dict) -> list:
    """
    Pre-sort predictions per entity so threshold evaluation is efficient.
    """
    curves = []

    for sid in entity_ids:
        true_set = gt_map.get(sid, set())
        preds = pred_map.get(sid, [])

        if not preds:
            curves.append((len(true_set), np.array([], dtype=float), np.array([], dtype=int)))
            continue

        preds = sorted(preds, key=lambda item: item[1], reverse=True)

        probs = np.array([p for _, p in preds], dtype=float)
        neg_probs = -probs

        tp_flags = [1 if cid in true_set else 0 for cid, _ in preds]
        cumulative_tp = np.cumsum(tp_flags, dtype=int)

        curves.append((len(true_set), neg_probs, cumulative_tp))

    return curves


def evaluate_threshold_curves(curves: list, threshold: float) -> float:
    scores = []
    neg_threshold = -threshold

    for true_len, neg_probs, cumulative_tp in curves:
        if neg_probs.size == 0:
            k = 0
        else:
            k = int(np.searchsorted(neg_probs, neg_threshold, side="right"))

        if true_len == 0:
            scores.append(1.0 if k == 0 else 0.0)
            continue

        if k == 0:
            scores.append(0.0)
            continue

        tp = int(cumulative_tp[k - 1])
        if tp == 0:
            scores.append(0.0)
            continue

        precision = tp / float(k)
        recall = tp / float(true_len)

        f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        scores.append(f05)

    if not scores:
        return 0.0

    return float(np.mean(scores))


def tune_threshold(
    entity_ids: list,
    pred_map: dict,
    gt_map: dict,
    default_threshold: float = 0.9,
) -> tuple:
    if not entity_ids:
        return default_threshold, 0.0

    curves = prepare_threshold_curves(entity_ids, pred_map, gt_map)

    thresholds = np.arange(0.01, 1.00, 0.01).tolist()
    thresholds += [0.995, 0.999]
    thresholds = sorted(set(thresholds))

    best_threshold = default_threshold
    best_score = -1.0

    for threshold in thresholds:
        score = evaluate_threshold_curves(curves, float(threshold))

        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and threshold > best_threshold
        ):
            best_threshold = float(threshold)
            best_score = score

    return best_threshold, best_score


# ------------------------------------------------------------------------------
# Output writers
# ------------------------------------------------------------------------------
def write_mapping_tsv(
    path: Path,
    header: list,
    entity_ids: list,
    mapping: dict,
) -> None:
    LOGGER.info("Writing %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(
            f,
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
            lineterminator="\n",
        )
        writer.writerow(header)

        for sid in entity_ids:
            ids = mapping.get(sid, [])
            ids = sorted(set(ids))
            writer.writerow([sid, ",".join(ids)])


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entity resolution pipeline for Amazon ML Challenge."
    )

    parser.add_argument("--data-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)

    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--max-candidates", type=int, default=160)

    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--disable-token-blocking", action="store_true")
    parser.add_argument(
        "--no-augment-positives",
        action="store_true",
        help="Do not add known training true positives to training candidate pairs.",
    )

    parser.add_argument("--nn-jobs", type=int, default=-1)

    parser.add_argument("--exact-key-limit", type=int, default=200)
    parser.add_argument("--exact-per-key", type=int, default=50)

    parser.add_argument("--token-df-limit", type=int, default=300)
    parser.add_argument("--token-ids-per-token", type=int, default=100)

    args = parser.parse_args()

    if args.max_candidates <= 0:
        args.max_candidates = args.top_k

    args.max_candidates = max(args.max_candidates, args.top_k)

    return args


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    # ------------------------------------------------------------------
    # Load training data
    # ------------------------------------------------------------------
    train_s1 = read_source(train_dir / "train_source1.tsv", "S1-")
    train_s2 = read_source(train_dir / "train_source2.tsv", "S2-")
    train_s3 = read_source(train_dir / "train_source3.tsv", "S3-")

    train_s1 = preprocess_source_df(train_s1)
    train_s2 = preprocess_source_df(train_s2)
    train_s3 = preprocess_source_df(train_s3)

    train_pool = pd.concat([train_s2, train_s3], ignore_index=True)

    train_s1_map = make_record_map(train_s1)
    train_pool_map = make_record_map(train_pool)

    all_train_s1_ids = train_s1["entity_id"].tolist()

    gt_map = read_ground_truth(
        train_dir / "train_ground_truth.tsv",
        all_train_s1_ids,
    )

    # ------------------------------------------------------------------
    # Validation split by Source 1 entity
    # ------------------------------------------------------------------
    if args.validation_size <= 0.0:
        train_ids = all_train_s1_ids
        val_ids = []
    elif len(all_train_s1_ids) >= 5:
        train_ids, val_ids = train_test_split(
            all_train_s1_ids,
            test_size=args.validation_size,
            random_state=args.seed,
            shuffle=True,
        )
        train_ids = list(train_ids)
        val_ids = list(val_ids)
    else:
        train_ids = all_train_s1_ids
        val_ids = []

    LOGGER.info(
        "Train Source1 entities=%d, validation Source1 entities=%d",
        len(train_ids),
        len(val_ids),
    )

    # ------------------------------------------------------------------
    # Generate training candidates
    # ------------------------------------------------------------------
    LOGGER.info("Generating training candidates")
    train_candidates = generate_candidates(train_s1, train_pool, args)
    log_candidate_stats(train_candidates, "Train candidates")

    # ------------------------------------------------------------------
    # Build train and validation feature tables
    # ------------------------------------------------------------------
    train_df = build_feature_dataframe(
        train_ids,
        train_candidates,
        gt_map,
        train_s1_map,
        train_pool_map,
        add_true_matches=(not args.no_augment_positives),
    )

    val_df = build_feature_dataframe(
        val_ids,
        train_candidates,
        gt_map,
        train_s1_map,
        train_pool_map,
        add_true_matches=False,
    )

    LOGGER.info("Train feature rows=%d", len(train_df))
    LOGGER.info("Validation feature rows=%d", len(val_df))

    # ------------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------------
    model = train_model(train_df)

    # ------------------------------------------------------------------
    # Threshold tuning on validation
    # ------------------------------------------------------------------
    if not val_df.empty:
        val_df["prob"] = predict_proba(model, val_df)
        val_pred_map = build_pred_map(val_df)
    else:
        val_pred_map = {}

    if args.threshold is not None:
        threshold = float(args.threshold)
        val_score = None
        LOGGER.info("Using supplied threshold=%s", threshold)
    else:
        threshold, val_score = tune_threshold(
            val_ids,
            val_pred_map,
            gt_map,
            default_threshold=0.9,
        )
        LOGGER.info("Tuned threshold=%s", threshold)
        LOGGER.info("Validation macro F_0.5=%s", val_score)

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    test_s1 = read_source(test_dir / "test_source1.tsv", "S1-")
    test_s2 = read_source(test_dir / "test_source2.tsv", "S2-")
    test_s3 = read_source(test_dir / "test_source3.tsv", "S3-")

    test_s1 = preprocess_source_df(test_s1)
    test_s2 = preprocess_source_df(test_s2)
    test_s3 = preprocess_source_df(test_s3)

    test_pool = pd.concat([test_s2, test_s3], ignore_index=True)

    test_s1_map = make_record_map(test_s1)
    test_pool_map = make_record_map(test_pool)

    test_ids = test_s1["entity_id"].tolist()

    # ------------------------------------------------------------------
    # Generate test candidates
    # ------------------------------------------------------------------
    LOGGER.info("Generating test candidates")
    test_candidates = generate_candidates(test_s1, test_pool, args)
    log_candidate_stats(test_candidates, "Test candidates")

    # ------------------------------------------------------------------
    # Score test candidates
    # ------------------------------------------------------------------
    test_df = build_feature_dataframe(
        test_ids,
        test_candidates,
        None,
        test_s1_map,
        test_pool_map,
        add_true_matches=False,
    )

    if not test_df.empty:
        test_df["prob"] = predict_proba(model, test_df)
        test_pred_map = build_pred_map(test_df)
    else:
        test_pred_map = {}

    # ------------------------------------------------------------------
    # Build final outputs
    # ------------------------------------------------------------------
    candidate_output = {}
    match_output = {}

    for sid in test_ids:
        candidate_ids = sorted(set(test_candidates.get(sid, {}).keys()))
        candidate_output[sid] = candidate_ids

        preds = test_pred_map.get(sid, [])
        selected = {cid for cid, prob in preds if prob >= threshold}

        # Safety check: final matches must be a subset of final candidates.
        candidate_set = set(candidate_ids)
        selected = selected.intersection(candidate_set)

        match_output[sid] = sorted(selected)

    write_mapping_tsv(
        output_dir / "candidate_pairs.tsv",
        ["source1_entity_id", "candidate_entity_ids"],
        test_ids,
        candidate_output,
    )

    write_mapping_tsv(
        output_dir / "matching_results.tsv",
        ["source1_entity_id", "matched_entity_ids"],
        test_ids,
        match_output,
    )

    match_counts = [len(v) for v in match_output.values()]
    LOGGER.info(
        "Test output entities=%d total_predicted_matches=%d entities_with_matches=%d",
        len(match_output),
        int(sum(match_counts)),
        int(sum(1 for c in match_counts if c > 0)),
    )
    LOGGER.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()